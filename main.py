import numpy as np
import os
import random
import shutil
import time
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sacred import Experiment, Ingredient
from sacred.observers import FileStorageObserver

from model import E2E
from data_reader.dataset_v1 import SpoofDatsetSystemID, SpoofDatsetEval
from src.eval_metrics import compute_eer
from src.loss import FocalLoss

ex  = Experiment('asvspoof19')
ex.observers.append(FileStorageObserver.create('snapshots'))

@ex.config
def my_config():
    model_params = {
        'MODEL_SELECT' : 1, # which model 
        'NUM_SPOOF_CLASS' : 7, # x-class classification
        'FOCAL_GAMMA' : None, # gamma parameter for focal loss; if obj is not focal loss, set this to None 
        'NUM_RESNET_BLOCK' : 5, # number of resnet blocks in ResNet 
        'AFN_UPSAMPLE' : 'Bilinear', # upsampling method in AFNet: Conv or Bilinear
        'AFN_ACTIVATION' : 'sigmoid', # activation function in AFNet: sigmoid, softmaxF, softmaxT
        'NUM_HEADS' : 3, # number of heads for multi-head att in SAFNet 
        'SAFN_HIDDEN' : 10, # hidden dim for SAFNet
        'SAFN_DIM' : 'T', # SAFNet attention dim: T or F
        'RNN_HIDDEN' : 128, # hidden dim for RNN
        'RNN_LAYERS' : 4, # number of hidden layers for RNN
        'RNN_BI': True, # bidirecitonal/unidirectional for RNN
        'DROPOUT_R' : 0.0, # dropout rate 
    }
    data_files = { # training 
        'train_scp': 'data_reader/feats/la_train_spec_tensor4.scp',
        'train_utt2index': 'data_reader/utt2systemID/la_train_utt2index_8',
        'dev_scp': 'data_reader/feats/la_dev_spec_tensor4.scp',
        'dev_utt2index': 'data_reader/utt2systemID/la_dev_utt2index_8',
        'dev_utt2systemID': 'data_reader/utt2systemID/la_dev_utt2systemID',
        'scoring_dir': 'scoring/la_cm_scores/',
    }
    
    leave_one_out = False # leave one out during train and val
    eer_criteria = False # train by dev acc or eer
    batch_size = 64
    test_batch_size = 64
    epochs = 10 # 20 for PA, 10 for LA 
    start_epoch = 1
    n_warmup_steps = 1000
    log_interval = 100
    pretrained = None # 'snapshots/119/model_best.pth.tar'
    pretrained_model_id = 181 # for forward pass 
    class_labels = [ # for post analysis 
        'bonafide', 'AB', 'AC', 
        'BA', 'BB', 'BC', 'CA', 'CB', 'CC', 'AA',
    ]

best_acc1 = 0
best_eer  = 10000

@ex.capture
def work(_run, pretrained, batch_size, test_batch_size, epochs, start_epoch, log_interval, n_warmup_steps, data_files, model_params, eer_criteria, leave_one_out):
    global best_acc1
    global best_eer
 
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tmp = torch.tensor([2]).to(device)

    # model is trained for binary classification (for datalaoder) 
    if model_params['NUM_SPOOF_CLASS'] == 2: 
        binary_class = True 
    else: binary_class = False 

    # model is RNN  
    if model_params['MODEL_SELECT'] == 4: 
        use_rnn = True 
    else: use_rnn = False 

    if model_params['FOCAL_GAMMA']: 
        print('training with focal loss')
        focal_obj = FocalLoss(gamma=model_params['FOCAL_GAMMA'])
    else: focal_obj = None

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    # create model
    # cnx
    model = E2E(**model_params).to(device) 
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print('===> Model total parameter: {}'.format(model_params))
    
    # Wrap model for multi-GPUs, if necessary
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print('multi-gpu') 
        model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    # cnx 
    optimizer = ScheduledOptim(
            torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        n_warmup_steps)

    # optionally resume from a checkpoint
    if pretrained:
        if os.path.isfile(pretrained):
            print("===> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained)
            start_epoch = checkpoint['epoch']
            if eer_criteria: best_eer = checkpoint['best_eer']
            else: best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("===> loaded checkpoint '{}' (epoch {})"
                  .format(pretrained, checkpoint['epoch']))
        else:
            print("===> no checkpoint found at '{}'".format(pretrained))

    cudnn.benchmark = True # It enables benchmark mode in cudnn.

    # Data loading code
    train_data = SpoofDatsetSystemID(data_files['train_scp'], data_files['train_utt2index'], binary_class, leave_one_out)
    val_data   = SpoofDatsetSystemID(data_files['dev_scp'], data_files['dev_utt2index'], binary_class, leave_one_out)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    if leave_one_out:
        eval_data   = SpoofDatsetSystemID(data_files['dev_scp'], data_files['dev_utt2index'], binary_class, False)
        eval_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    else: eval_loader = val_loader

    best_epoch = 0
    early_stopping, max_patience = 0, 5 # for early stopping 
    for epoch in range(start_epoch, start_epoch+epochs):

        # train for one epoch
        train(train_loader, model, optimizer, epoch, device, log_interval, use_rnn, focal_obj)

        # evaluate on validation set
        acc1 = validate(val_loader, data_files['dev_utt2systemID'], model, device, log_interval, 
                use_rnn, eer_criteria, focal_obj)

        # remember best acc@1/eer and save checkpoint
        if eer_criteria:
            is_best = acc1 < best_eer
            best_eer = min(acc1, best_eer)
        else:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        
        # adjust learning rate + early stopping 
        if is_best:
            early_stopping = 0
            best_epoch = epoch + 1
        else:
            early_stopping += 1
            if epoch - best_epoch > 2:
                optimizer.increase_delta()
                best_epoch = epoch + 1
        if early_stopping == max_patience:
            break
        
        # save model
        if not is_best:
            continue
        if eer_criteria:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_eer': best_eer,
                'optimizer' : optimizer.state_dict(),
            }, is_best,  "snapshots/" + str(_run._id), str(epoch) + ('_%.3f'%acc1) + ".pth.tar")
        else:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best,  "snapshots/" + str(_run._id), str(epoch) + ('_%.3f'%acc1) + ".pth.tar")

    # load best model 
    best_model_pth = os.path.join('snapshots', str(_run._id), 'model_best.pth.tar')
    score_file_pth = os.path.join(data_files['scoring_dir'], str(_run._id) + '-scores.txt')
    print("===> loading best model for scoring: '{}'".format(best_model_pth))
    checkpoint = torch.load(best_model_pth)
    model.load_state_dict(checkpoint['state_dict'])

    # compute EER 
    print("===> scoring file saved at: '{}'".format(score_file_pth))
    prediction(eval_loader, model, device, score_file_pth, data_files['dev_utt2systemID'], use_rnn, focal_obj)


@ex.capture
def post(_run, pretrained, test_batch_size, data_files, model_params, eer_criteria, class_labels):
    """ what are the classes that performed well, and the classes that did not perform well """
    
    use_cuda = torch.cuda.is_available() # use cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    tmp = torch.tensor([2]).to(device)

    # model is RNN  
    if model_params['MODEL_SELECT'] == 4: 
        use_rnn = True 
    else: use_rnn = False 

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    # create model
    # cnx
    model = E2E(**model_params).to(device) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print('===> Model total parameter: {}'.format(num_params))
    
    if pretrained:
        if os.path.isfile(pretrained):
            print("===> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage) # load for cpu
            if eer_criteria: best_eer = checkpoint['best_eer']
            else: best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("===> loaded checkpoint '{}' (epoch {})"
                  .format(pretrained, checkpoint['epoch']))
        else:
            print("===> no checkpoint found at '{}'".format(pretrained))

    # Data loading code (class analysis for multi-class classification only)
    train_data = SpoofDatsetSystemID(data_files['train_scp'], data_files['train_utt2index'], binary_class=False)
    val_data   = SpoofDatsetSystemID(data_files['dev_scp'], data_files['dev_utt2index'], binary_class=False)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    
    # class analysis for train and dev 
    print("===> class analysis for train set")
    class_analysis(train_loader, model, device, use_rnn, class_labels, use_focal=model_params['FOCAL_GAMMA'])
    print("===> class analysis for dev set")
    class_analysis(val_loader, model, device, use_rnn, class_labels, use_focal=model_params['FOCAL_GAMMA'])


@ex.capture
def forward_pass(_run, pretrained_model_id, test_batch_size, data_files, model_params, eer_criteria, class_labels):
    """ forward pass dev and eval data to trained model  """
    
    use_cuda = torch.cuda.is_available() # use cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    tmp = torch.tensor([2]).to(device)

    # model is RNN  
    if model_params['MODEL_SELECT'] == 4: 
        use_rnn = True 
    else: use_rnn = False 
    
    # model is trained with focal loss objective 
    if model_params['FOCAL_GAMMA']: 
        print('training with focal loss')
        focal_obj = FocalLoss(gamma=model_params['FOCAL_GAMMA'])
    else: focal_obj = None

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    # create model
    # cnx
    model = E2E(**model_params).to(device) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print('===> Model total parameter: {}'.format(num_params))
    
    if pretrained_model_id:
        pretrain_pth = 'snapshots/' + str(pretrained_model_id) + '/model_best.pth.tar'
        if os.path.isfile(pretrain_pth):
            print("===> loading checkpoint '{}'".format(pretrain_pth))
            checkpoint = torch.load(pretrain_pth, map_location=lambda storage, loc: storage) # load for cpu
            if eer_criteria: best_eer = checkpoint['best_eer']
            else: best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("===> loaded checkpoint '{}' (epoch {})"
                  .format(pretrain_pth, checkpoint['epoch']))
        else:
            print("===> no checkpoint found at '{}'".format(pretrain_pth))
            exit() 

    # Data loading code (class analysis for multi-class classification only)
    val_data    = SpoofDatsetSystemID(data_files['dev_scp'], data_files['dev_utt2index'], binary_class=False)
    #val_data    = SpoofDatsetEval(data_files['dev_scp'])
    eval_data   = SpoofDatsetEval(data_files['eval_scp'])
    val_loader  = torch.utils.data.DataLoader(
        val_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=test_batch_size, shuffle=False, **kwargs)

    # forward pass for dev
    print("===> forward pass for dev set")
    score_file_pth = os.path.join(data_files['scoring_dir'], str(pretrained_model_id) + '-dev_scores.txt')
    print("===> dev scoring file saved at: '{}'".format(score_file_pth))
    prediction(val_loader, model, device, score_file_pth, data_files['dev_utt2systemID'], use_rnn, focal_obj)
    # forward pass for eval
    print("===> forward pass for eval set")
    score_file_pth = os.path.join(data_files['scoring_dir'], str(pretrained_model_id) + '-eval_scores.txt')
    print("===> eval scoring file saved at: '{}'".format(score_file_pth))
    eval_prediction(eval_loader, model, device, score_file_pth, data_files['eval_utt2systemID'], use_rnn, focal_obj)
 

def train(train_loader, model, optimizer, epoch, device, log_interval, rnn, focal_obj):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (_, input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # Create vaiables
        input  = input.to(device, non_blocking=True)
        #input  = input.to(device)
        target = target.to(device, non_blocking=True).view((-1,))
        #target = target.to(device).view((-1,))

        # compute output
        if rnn: 
            hidden = model.init_hidden(input.size(0))
            output = model(input, hidden)
        else: output = model(input)

        # loss 
        if focal_obj: loss = focal_obj(output, target)
        else: loss = F.nll_loss(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.6f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lr=lr, loss=losses, top1=top1))


def validate(val_loader, utt2systemID_file, model, device, log_interval, rnn, eer, focal_obj):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    utt2scores = defaultdict(list)
    with torch.no_grad():
        end = time.time()
        for i, (utt_list, input, target) in enumerate(val_loader):
            input  = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).view((-1,))

            # compute output
            if rnn: 
                hidden = model.init_hidden(input.size(0))
                output = model(input, hidden)
            else: output = model(input)
    
            # loss 
            if focal_obj: loss = focal_obj(output, target)
            else: loss = F.nll_loss(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # for model selection criteria == dev_eer only 
            if eer:
                score  = output[:,0] # use log-probability of the bonafide class for scoring 
                for index, utt_id in enumerate(utt_list):
                    curr_utt = ''.join(utt_id.split('-')[0] + '-' + utt_id.split('-')[1])
                    utt2scores[curr_utt].append(score[index].item()) 
   
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

    print('===> Acc@1 {top1.avg:.3f}\n'.format(top1=top1))
    
    if eer:     
        # first do averaging
        with open(utt2systemID_file, 'r') as f:
            temp = f.readlines()
        content  = [x.strip() for x in temp]
        utt_list = [x.split()[0] for x in content]
        id_list  = [x.split()[1] for x in content] 

        spoof_cm, bona_cm  = [], []
        for index, utt_id in enumerate(utt_list):
            if utt_id not in utt2scores.keys(): # condition for leave on out 
                continue 
            score_list = utt2scores[utt_id]   
            avg_score  = reduce(lambda x, y: x + y, score_list) / len(score_list)
            spoof_id = id_list[index]
            if spoof_id == 'bonafide':
                bona_cm.append(avg_score)
            else: spoof_cm.append(avg_score)
       
        spoof_cm, bona_cm = np.array(spoof_cm), np.array(bona_cm)
        eer_cm = 100.*compute_eer(bona_cm, spoof_cm)[0]
    
        print('===> EER_CM: {}\n'.format(eer_cm))
    
    if eer: return eer_cm
    else: return top1.avg


def prediction(val_loader, model, device, output_file, utt2systemID_file, rnn, focal_obj):
    
    # switch to evaluate mode
    utt2scores = defaultdict(list) 
    model.eval()

    with torch.no_grad():
        for i, (utt_list, input, target) in enumerate(val_loader):
            input  = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).view((-1,))

            # compute output
            if rnn: 
                hidden = model.init_hidden(input.size(0))
                output = model(input, hidden)
            else: output = model(input)

            # get score 
            if focal_obj: 
                output = F.log_softmax(output, dim=-1) # apply softmax if model trained with focal loss 
            score = output[:,0] # use log-probability of the bonafide class for scoring 
            
            for index, utt_id in enumerate(utt_list):
                curr_utt = ''.join(utt_id.split('-')[0] + '-' + utt_id.split('-')[1])
                utt2scores[curr_utt].append(score[index].item()) 
   
        # first do averaging
        with open(utt2systemID_file, 'r') as f:
            temp = f.readlines()
        content  = [x.strip() for x in temp]
        utt_list = [x.split()[0] for x in content]
        id_list  = [x.split()[1] for x in content] 

        with open(output_file, 'w') as f:
            for index, utt_id in enumerate(utt_list):
                score_list = utt2scores[utt_id]   
                avg_score  = reduce(lambda x, y: x + y, score_list) / len(score_list)
                spoof_id = id_list[index]
                if spoof_id == 'bonafide':
                    f.write('%s %s %s %f\n' % (utt_id, '-', 'bonafide', avg_score))
                else: 
                    f.write('%s %s %s %f\n' % (utt_id, spoof_id, 'spoof', avg_score))


def eval_prediction(eval_loader, model, device, output_file, utt2systemID_file, rnn, focal_obj):
    """ eval data's utt2systemID does not have ID """ 

    # switch to evaluate mode
    utt2scores = defaultdict(list) 
    model.eval()

    with torch.no_grad():
        for i, (utt_list, input) in enumerate(eval_loader):
            input  = input.to(device, non_blocking=True)

            # compute output
            if rnn: 
                hidden = model.init_hidden(input.size(0))
                output = model(input, hidden)
            else: output = model(input)

            # get score 
            if focal_obj: 
                output = F.log_softmax(output, dim=-1) # apply softmax if model trained with focal loss 
            score = output[:,0] # use log-probability of the bonafide class for scoring 
            
            for index, utt_id in enumerate(utt_list):
                curr_utt = ''.join(utt_id.split('-')[0] + '-' + utt_id.split('-')[1])
                utt2scores[curr_utt].append(score[index].item()) 
   
        # first do averaging
        with open(utt2systemID_file, 'r') as f:
            temp = f.readlines()
        content  = [x.strip() for x in temp]
        utt_list = content 

        with open(output_file, 'w') as f:
            for index, utt_id in enumerate(utt_list):
                score_list = utt2scores[utt_id]   
                avg_score  = reduce(lambda x, y: x + y, score_list) / len(score_list)
                f.write('%s %f\n' % (utt_id, avg_score))


def class_analysis(val_loader, model, device, rnn, class_labels, use_focal):

    model.eval()
    
    # for recording acc. for each class
    class_correct = list(0. for i in range(len(class_labels)))
    class_total = list(0. for i in range(len(class_labels)))
    
    with torch.no_grad():
        for i, (utt_list, input, target) in enumerate(val_loader):
            input  = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).view((-1,))

            # compute output
            if rnn: 
                hidden = model.init_hidden(input.size(0))
                output = model(input, hidden)
            else: output = model(input)
            
            # get score 
            if use_focal: 
                output = F.log_softmax(output, dim=-1) # apply softmax if model trained with focal loss 

            # measure accuracy 
            _, predicted = torch.max(output, 1) 
            c = (predicted == target).squeeze()
            
            # record results
            for i in range(input.size(0)):
                label = target[i] # get which class 
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(class_labels)):
        print("===> Results for each class")
        print('Accuracy of %8s : %6f %%' % (class_labels[i], 
            100. * class_correct[i] / class_total[i]))


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, path + '/' + filename)
    if is_best:
        print("===> save to checkpoint at {}\n".format(path + '/' + 'model_best.pth.tar'))
        shutil.copyfile(path + '/' + filename, path + '/' + 'model_best.pth.tar')


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])


@ex.automain
def main():
    work()
    #post()
    #forward_pass()
