import numpy as np
import torch
from torch.utils import data
from kaldi_io import read_mat
import h5py

# PyTorch Dataset 

class SpoofDatsetEval(data.Dataset):
    ''' Evaluation, no label
    '''
    def __init__(self, scp_file):
        with open(scp_file) as f:
            temp = f.readlines()
        content = [x.strip() for x in temp]
        self.key_dic = {index: i.split()[0] for (index, i) in enumerate(content)}
        self.ark_dic = {index: i.split()[1] for (index, i) in enumerate(content)}

    def __len__(self):
        return len(self.key_dic.keys())

    def __getitem__(self, index):
        utt_id = self.key_dic[index]
        X = np.expand_dims(read_mat(self.ark_dic[index]), axis=0)

        return utt_id, X


class SpoofLeaveOneOutDatset(data.Dataset):
    '''
    Leave out 
        AA (for PA) 
        SS_1 (for LA)
    during training, to test how NN generalizes to new attack condition
    
    classification label becomes: 
        multi-class classification for PA: AA, AB, AC, BA, BB, BC, CA, CB, CC --> 10 classes
        (bonafide: 0), (AB: 1), (AC: 2), (BA: 3), (BB: 4), (BC: 5),
        (CA: 6), (CB: 7), (CC: 8) +- (AA:9)

        multi-class classification for LA: SS_1, SS_2, SS_4, US_1, VC_1, VC_4 --> 7 classes
        (bonafide: 0), (SS_2: 1), (SS_4: 2), (US_1: 3), (VC_1: 4), (VC_4: 5) +- (SS_1: 6)
    '''
    def __init__(self, scp_file, utt2index_file, mode='train', condition='PA'):
        with open(scp_file) as f:
            temp = f.readlines()
        content = [x.strip() for x in temp]
        self.key_dic = {index: i.split()[0] for (index, i) in enumerate(content)}
        self.ark_dic = {index: i.split()[1] for (index, i) in enumerate(content)}

        with open(utt2index_file) as f:
            temp = f.readlines()
        self.label_dic = {index: int(x.strip().split()[1]) for (index, x) in enumerate(temp)}
         
        for index,label in self.label_dic.items():
            if label == 1:
                if mode == 'train': # remove label AA (for PA) or SS_1 (for LA)
                    self.key_dic.pop(index)
                elif mode == 'test':
                    if condition == 'PA':
                        self.label_dic[index] = 9
                    elif condition == 'LA':
                        self.label_dic[index] = 6
            if label > 1: 
                self.label_dic[index] = label-1
        counter = 0 
        self.mapping = {}
        for index in self.key_dic.keys(): # because of the popping, indexing is messed up
            self.mapping[counter] = index 
            counter += 1
        
    def __len__(self):
        return len(self.mapping.keys())

    def __getitem__(self, counter):
        index = self.mapping[counter]
        utt_id = self.key_dic[index]
        X = np.expand_dims(read_mat(self.ark_dic[index]), axis=0)
        y = self.label_dic[index]

        return utt_id, X, y


class SpoofDatsetSystemID3(data.Dataset):
    '''
    use hdf5 file instead of ark file to access feats 
    '''
    def __init__(self, raw, scp_file, utt2index_file):
        self.h5f = h5py.File(raw, 'r')
        with open(scp_file) as f:
            temp = f.readlines()
        content = [x.strip() for x in temp]
        self.key_dic = {index: i.split()[0] for (index, i) in enumerate(content)}

        with open(utt2index_file) as f:
            temp = f.readlines()
        self.label_dic = {index: int(x.strip().split()[1]) for (index, x) in enumerate(temp)}

        assert len(self.key_dic.keys()) == len(self.label_dic.keys())
    
    def __len__(self):
        return len(self.key_dic.keys())

    def __getitem__(self, index):
        utt_id = self.key_dic[index]
        X = np.expand_dims(self.h5f[utt_id][:], axis=0)
        y = self.label_dic[index]

        return utt_id, X, y


class SpoofDatsetSystemID2(data.Dataset):
    '''
    read all data onto the disc instead of reading it on the fly 
    '''
    def __init__(self, scp_file, utt2index_file):
        with open(scp_file) as f:
            temp = f.readlines()
        content = [x.strip() for x in temp]
        self.key_dic = {index: i.split()[0] for (index, i) in enumerate(content)}
        self.feat_dic = {index: np.expand_dims(read_mat(i.split()[1]), axis=0) 
                for (index, i) in enumerate(content)}

        with open(utt2index_file) as f:
            temp = f.readlines()
        self.label_dic = {index: int(x.strip().split()[1]) for (index, x) in enumerate(temp)}

        assert len(self.key_dic.keys()) == len(self.label_dic.keys())
    
    def __len__(self):
        return len(self.key_dic.keys())

    def __getitem__(self, index):
        utt_id = self.key_dic[index]
        X = self.feat_dic[index]
        y = self.label_dic[index]

        return utt_id, X, y


class SpoofDatsetSystemID(data.Dataset):
    ''' multi-class classification for PA: AA, AB, AC, BA, BB, BC, CA, CB, CC --> 10 classes
        (bonafide: 0), (AA: 1), (AB: 2), (AC: 3), (BA: 4), (BB: 5), (BC: 6),
        (CA: 7), (CB: 8), (CC: 9)

        multi-class classification for LA: SS_1, SS_2, SS_4, US_1, VC_1, VC_4 --> 7 classes
        (bonafide: 0), (SS_1: 1), (SS_2: 2), (SS_4: 3), (US_1: 4), (VC_1: 5), (VC_4: 6)

        if leave_one_out: 
            for pa: leave out the class with label == 9 
            for la: leave out the class with label == 6
    '''
    def __init__(self, scp_file, utt2index_file, binary_class, leave_one_out=False):
        with open(scp_file) as f:
            temp = f.readlines()
        content = [x.strip() for x in temp]
        self.key_dic = {index: i.split()[0] for (index, i) in enumerate(content)}
        self.ark_dic = {index: i.split()[1] for (index, i) in enumerate(content)}

        with open(utt2index_file) as f:
            temp = f.readlines()
        temp_dic = {index: int(x.strip().split()[1]) for (index, x) in enumerate(temp)}

        # leave one out 
        self.all_idx = {}
        counter = 0 
        for index, label in temp_dic.items():
            if leave_one_out:
                if label != 6: 
                    self.all_idx[counter] = index
                    counter += 1
            else: 
                self.all_idx[counter] = index 
                counter += 1
        
        if binary_class:
            self.label_dic = {index: 0 if orig_label == 0 else 1 for (index, orig_label) in temp_dic.items()}
        else: self.label_dic = temp_dic
        
        if not leave_one_out:
            assert len(self.all_idx.keys()) == len(self.label_dic.keys())

    def __len__(self):
        return len(self.all_idx.keys())

    def __getitem__(self, counter):
        index = self.all_idx[counter]
        utt_id = self.key_dic[index]
        X = np.expand_dims(read_mat(self.ark_dic[index]), axis=0)
        y = self.label_dic[index]

        return utt_id, X, y
