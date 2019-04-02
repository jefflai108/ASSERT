from __future__ import print_function
import os
import numpy as np
import kaldi_io as ko

# IMPORTANT: run this with python3 (don't use Nanxin's environment)
# Python implementation of Unified Feature Map 

def tensor_cnn_utt(mat, truncate_len):
    mat = np.swapaxes(mat, 0, 1)
    max_len = truncate_len * int(np.ceil(mat.shape[1]/truncate_len))
    repetition = int(max_len/mat.shape[1])
    tensor = np.tile(mat,repetition)
    repetition = max_len % mat.shape[1]
    rest = mat[:,:repetition]
    tensor = np.hstack((tensor,rest))
    
    return tensor


def construct_tensor(orig_feat_scp, ark_scp_output, tuncate_len):
    with ko.open_or_fd(ark_scp_output, 'wb') as f:
        for key,mat in ko.read_mat_scp(orig_feat_scp):
            tensor = tensor_cnn_utt(mat, truncate_len)
            repetition = int(tensor.shape[1]/truncate_len)
            for i in range(repetition):
                sub_tensor = tensor[:,i*truncate_len:(i+1)*truncate_len]
                new_key = key + '-' + str(i)
                ko.write_mat(f, sub_tensor, key=new_key)


def construct_slide_tensor(orig_feat_scp, ark_scp_output, tuncate_len):
    with ko.open_or_fd(ark_scp_output, 'wb') as f:
        for key,mat in ko.read_mat_scp(orig_feat_scp):
            tensor = tensor_cnn_utt(mat, truncate_len)
            repetition = int(tensor.shape[1]/truncate_len)
            repetition = 2 * repetition - 1 # slide 
            for i in range(repetition):
                sub_tensor = tensor[:,200*i:200*i+truncate_len]
                new_key = key + '-' + str(i)
                ko.write_mat(f, sub_tensor, key=new_key)


if __name__ == '__main__':
    curr_wd = os.getcwd()
    """
    for mode in ['la_dev_', 'la_train_']:
        truncate_len = 400
        for feat in ['spec_cm_', 'spec_']:
            in_scp  = curr_wd + '/' + 'feats/' + mode + feat + 'orig.scp'
            out_scp = curr_wd + '/' + 'feats/' + mode + feat + 'tensor3.scp'
            out_ark = curr_wd + '/' + 'feats/' + mode + feat + 'tensor3.ark'
            ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_ark + ',' + out_scp
            construct_tensor(in_scp, ark_scp, truncate_len)
    
    for mode in ['pa_train_', 'pa_dev_']:
        truncate_len = 500
        for feat in ['spec_', 'spec_cm_']:
            in_scp  = curr_wd + '/' + 'feats/' + mode + feat + 'orig.scp'
            out_scp = curr_wd + '/' + 'feats/' + mode + feat + 'tensor.scp'
            out_ark = curr_wd + '/' + 'feats/' + mode + feat + 'tensor.ark'
            ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_ark + ',' + out_scp
            construct_tensor(in_scp, ark_scp, truncate_len)
    
    for mode in ['pa_train_', 'pa_dev_']:
        truncate_len = 400
        for feat in ['fbank_cm_', 'fbank_']:
            in_scp  = curr_wd + '/' + 'feats/' + mode + feat + 'orig.scp'
            out_scp = curr_wd + '/' + 'feats/' + mode + feat + 'tensor3.scp'
            out_ark = curr_wd + '/' + 'feats/' + mode + feat + 'tensor3.ark'
            ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_ark + ',' + out_scp
            construct_tensor(in_scp, ark_scp, truncate_len)
    
    for mode in ['la_dev_', 'la_train_']:
        truncate_len = 400
        for feat in ['spec_']:
            in_scp  = curr_wd + '/' + 'feats/' + mode + feat + 'orig.scp'
            out_scp = curr_wd + '/' + 'feats/' + mode + feat + 'tensor4.scp'
            out_ark = curr_wd + '/' + 'feats/' + mode + feat + 'tensor4.ark'
            ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_ark + ',' + out_scp
            construct_slide_tensor(in_scp, ark_scp, truncate_len)
    
    for mode in ['pa_train_', 'pa_dev_']:
        truncate_len = 400
        for feat in ['spec_']:
            in_scp  = curr_wd + '/' + 'feats/' + mode + feat + 'orig.scp'
            out_scp = curr_wd + '/' + 'feats/' + mode + feat + 'tensor4.scp'
            out_ark = curr_wd + '/' + 'feats/' + mode + feat + 'tensor4.ark'
            ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_ark + ',' + out_scp
            construct_slide_tensor(in_scp, ark_scp, truncate_len)
    """
    for mode in ['la_eval_', 'pa_eval_']:
        truncate_len = 400
        for feat in ['spec_', 'spec_cm_']:
            in_scp  = curr_wd + '/' + 'feats/' + mode + feat + 'orig.scp'
            out_scp = curr_wd + '/' + 'feats/' + mode + feat + 'tensor1.scp'
            out_ark = curr_wd + '/' + 'feats/' + mode + feat + 'tensor1.ark'
            ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_ark + ',' + out_scp
            construct_tensor(in_scp, ark_scp, truncate_len)
    
    for mode in ['la_eval_', 'pa_eval_']:
        truncate_len = 400
        for feat in ['spec_', 'spec_cm_']:
            in_scp  = curr_wd + '/' + 'feats/' + mode + feat + 'orig.scp'
            out_scp = curr_wd + '/' + 'feats/' + mode + feat + 'tensor2.scp'
            out_ark = curr_wd + '/' + 'feats/' + mode + feat + 'tensor2.ark'
            ark_scp = 'ark:| copy-feats --compress=true ark:- ark,scp:' + out_ark + ',' + out_scp
            construct_slide_tensor(in_scp, ark_scp, truncate_len)

