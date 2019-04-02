from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
from src import attentive_filtering_network, resnet, senet

def E2E(MODEL_SELECT, NUM_SPOOF_CLASS, NUM_RESNET_BLOCK, AFN_UPSAMPLE, AFN_ACTIVATION, NUM_HEADS, 
        SAFN_HIDDEN, SAFN_DIM, RNN_HIDDEN, RNN_LAYERS, DROPOUT_R, RNN_BI, FOCAL_GAMMA):
    if FOCAL_GAMMA: FOCAL_LOSS = True 
    else: FOCAL_LOSS = False 

    if MODEL_SELECT == 1:
        print('resnet')
        model = resnet.SpoofSmallResNet257_400(NUM_SPOOF_CLASS, NUM_RESNET_BLOCK, FOCAL_LOSS)
    elif MODEL_SELECT == 5:
        print('attentive filtering network')
        model = attentive_filtering_network.SpoofSmallAFNet257_400(NUM_SPOOF_CLASS, AFN_UPSAMPLE, AFN_ACTIVATION, NUM_RESNET_BLOCK, FOCAL_LOSS)
    elif MODEL_SELECT == 6:
        print('squeeze-and-excitation network')
        #model = senet.se_resnet18(num_classes=NUM_SPOOF_CLASS, focal_loss=FOCAL_LOSS)
        #model = senet.se_resnet34(num_classes=NUM_SPOOF_CLASS, focal_loss=FOCAL_LOSS)
        model = senet.se_resnet50(num_classes=NUM_SPOOF_CLASS, focal_loss=FOCAL_LOSS)
        #model = senet.se_resnet101(num_classes=NUM_SPOOF_CLASS, focal_loss=FOCAL_LOSS)
        #model = senet.se_resnet152(num_classes=NUM_SPOOF_CLASS, focal_loss=FOCAL_LOSS)

    return model 

def test_E2E():
    model_params = {
        'MODEL_SELECT' : 3, # which model 
        'NUM_SPOOF_CLASS' : 10, # x-class classification
        'FOCAL_GAMMA': None, # gamma parameter for focal loss; if obj is not focal loss, set this to None
        'NUM_RESNET_BLOCK' : 5, # number of resnet blocks in ResNet 
        'AFN_UPSAMPLE' : 'Bilinear', # upsampling method in AFNet: Conv or Bilinear
        'AFN_ACTIVATION' : 'sigmoid', # activation function in AFNet: sigmoid, softmaxF, softmaxT
        'NUM_HEADS' : 3, # number of heads for multi-head att in SAFNet 
        'SAFN_HIDDEN' : 10, # hidden dim for SAFNet
        'SAFN_DIM' : 'T', # SAFNet attention dim: T or F
        'RNN_HIDDEN' : 128, # hidden dim for RNN
        'RNN_LAYERS' : 4, # number of hidden layers for RNN
        'RNN_BI' : True, # bidirection/unidirection for RNN
        'DROPOUT_R' : 0.0, # dropout rate 
    }
    model = E2E(**model_params)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model contains {} parameters'.format(model_params))
    print(model)
    x = torch.randn(2,1,257,400)
    output = model(x)
    #print(output)
    
if __name__ == '__main__':
    test_E2E()
