
# create multi-labels for model training 

def convert_la(scp_file, systemID_file, out_file):
    ''' multi-class classification for LA: SS_1, SS_2, SS_4, US_1, VC_1, VC_4 --> 7 classes
        (bonafide: 0), (SS_1: 1), (SS_2: 2), (SS_4: 3), (US_1: 4), (VC_1: 5), (VC_4: 6)
    '''
    with open(scp_file) as f:
        temp = f.readlines()
    key_list = [x.strip().split()[0] for x in temp]

    with open(systemID_file) as f:
        temp = f.readlines()
    utt2ID = {x.strip().split()[0]:x.strip().split()[1] for x in temp}

    with open(out_file, 'w') as f:
        for key in key_list:
            curr_utt = ''.join(key.split('-')[0] + '-' + key.split('-')[1])
            label = utt2ID[curr_utt] 
            if label == 'bonafide':
                f.write('%s %d\n' % (key, 0))
            elif label == 'SS_1':
                f.write('%s %d\n' % (key, 1))
            elif label == 'SS_2':
                f.write('%s %d\n' % (key, 2))
            elif label == 'SS_4':
                f.write('%s %d\n' % (key, 3))
            elif label == 'US_1':
                f.write('%s %d\n' % (key, 4))
            elif label == 'VC_1':
                f.write('%s %d\n' % (key, 5))
            elif label == 'VC_4':
                f.write('%s %d\n' % (key, 6))


def convert_pa(scp_file, systemID_file, out_file):
    ''' multi-class classification for PA: AA, AB, AC, BA, BB, BC, CA, CB, CC --> 10 classes
        (bonafide: 0), (AA: 1), (AB: 2), (AC: 3), (BA: 4), (BB: 5), (BC: 6),
        (CA: 7), (CB: 8), (CC: 9)
    '''
    with open(scp_file) as f:
        temp = f.readlines()
    key_list = [x.strip().split()[0] for x in temp]

    with open(systemID_file) as f:
        temp = f.readlines()
    utt2ID = {x.strip().split()[0]:x.strip().split()[1] for x in temp}

    with open(out_file, 'w') as f:
        for key in key_list:
            curr_utt = ''.join(key.split('-')[0] + '-' + key.split('-')[1])
            label = utt2ID[curr_utt] 
            if label == 'bonafide':
                f.write('%s %d\n' % (key, 0))
            elif label == 'AA':
                f.write('%s %d\n' % (key, 1))
            elif label == 'AB':
                f.write('%s %d\n' % (key, 2))
            elif label == 'AC':
                f.write('%s %d\n' % (key, 3))
            elif label == 'BA':
                f.write('%s %d\n' % (key, 4))
            elif label == 'BB':
                f.write('%s %d\n' % (key, 5))
            elif label == 'BC':
                f.write('%s %d\n' % (key, 6))
            elif label == 'CA':
                f.write('%s %d\n' % (key, 7))
            elif label == 'CB':
                f.write('%s %d\n' % (key, 8))
            elif label == 'CC':
                f.write('%s %d\n' % (key, 9))


def convert_pa_leave_one_out(scp_file, systemID_file, out_file):
    ''' multi-class classification for PA: AA, AB, AC, BA, BB, BC, CA, CB, CC --> 10 classes
        (bonafide: 0), (AB: 1), (AC: 2), (BA: 3), (BB: 4), (BC: 5),
        (CA: 6), (CB: 7), (CC: 8), (AA: 9)
    '''
    with open(scp_file) as f:
        temp = f.readlines()
    key_list = [x.strip().split()[0] for x in temp]

    with open(systemID_file) as f:
        temp = f.readlines()
    utt2ID = {x.strip().split()[0]: x.strip().split()[1] for x in temp}

    with open(out_file, 'w') as f:
        for key in key_list:
            curr_utt = ''.join(key.split('-')[0] + '-' + key.split('-')[1])
            label = utt2ID[curr_utt] 
            if label == 'bonafide':
                f.write('%s %d\n' % (key, 0))
            elif label == 'AA':
                f.write('%s %d\n' % (key, 9))
            elif label == 'AB':
                f.write('%s %d\n' % (key, 1))
            elif label == 'AC':
                f.write('%s %d\n' % (key, 2))
            elif label == 'BA':
                f.write('%s %d\n' % (key, 3))
            elif label == 'BB':
                f.write('%s %d\n' % (key, 4))
            elif label == 'BC':
                f.write('%s %d\n' % (key, 5))
            elif label == 'CA':
                f.write('%s %d\n' % (key, 6))
            elif label == 'CB':
                f.write('%s %d\n' % (key, 7))
            elif label == 'CC':
                f.write('%s %d\n' % (key, 8))


if __name__ == '__main__':
    curr_wd = 'utt2systemID/'
    systemID_files = ['la_dev_utt2systemID', 'la_train_utt2systemID',
            'pa_dev_utt2systemID', 'pa_train_utt2systemID']
    out_files = ['la_dev_utt2index_8', 'la_train_utt2index_8',
            'pa_dev_utt2index_8', 'pa_train_utt2index_8']
    scp_files = ['feats/la_dev_spec_tensor4.scp', 'feats/la_train_spec_tensor4.scp',
            'feats/pa_dev_spec_tensor4.scp', 'feats/pa_train_spec_tensor4.scp']

    #systemID_files = ['pa_dev_utt2systemID', 'pa_train_utt2systemID'] 
    #out_files = ['pa_dev_utt2index_6', 'pa_train_utt2index_6']
    #scp_files = ['feats/pa_dev_spec_tensor3.scp', 'feats/pa_train_spec_tensor3.scp']

    for i in range(0,2):
        convert_la(scp_files[i], curr_wd + systemID_files[i], 
                curr_wd + out_files[i])

    for i in range(2,4):
        convert_pa(scp_files[i], curr_wd + systemID_files[i], 
                curr_wd + out_files[i])

    """ 
    la  |  spec with cm  |  la_{train,dev}_spec_cm_tensor.scp   |  (257 by 400)  |  la_{train,dev}_utt2index    |  regular 
    pa  |  spec with cm  |  pa_{train,dev}_spec_cm_tensor.scp   |  (257 by 500)  |  pa_{train,dev}_utt2index    |  regular 
    pa  |  spec          |  pa_{train,dev}_spec_tensor3.scp     |  (257 by 400)  |  pa_{train,dev}_utt2index_2  |  regular
    pa  |  spec with cm  |  pa_{train,dev}_spec_cm_tensor2.scp  |  (257 by 400)  |  pa_{train,dev}_utt2index_2  |  regular
    la  |  cqcc with cm  |  la_{train,dev}_cqcc_cm_tensor.scp   |  (30  by 400)  |  la_{train,dev}_utt2index_3  |  regular
    pa  |  cqcc with cm  |  pa_{train,dev}_cqcc_cm_tensor2.scp  |  (30  by 400)  |  pa_{train,dev}_utt2index_4  |  regular
    la  |  cqcc          |  la_{train,dev}_cqcc_tensor.scp      |  (30  by 400)  |  la_{train,dev}_utt2index_3  |  regular
    pa  |  cqcc          |  pa_{train,dev}_cqcc_tensor2.scp     |  (30  by 400)  |  pa_{train,dev}_utt2index_4  |  regular
    pa  |  cpc_1         |  pa_{train,dev}_cpc_1_tensor_400.scp |  (256 by 400)  |  pa_{train,dev}_utt2index_5  |  regular
    pa  |  spec          |  pa_{train,dev}_spec_tensor3.scp     |  (257 by 400)  |  pa_{train,dev}_utt2index_6  |  leave out AA
    la  |  spec          |  la_{train,dev}_spec_tensor.scp      |  (257 by 400)  |  la_{train,dev}_utt2index    |  regular 
    la  |  cpc_2         |  la_{train,dev}_cpc_2_tensor_400.scp |  (257 by 400)  |  la_{train,dev}_utt2index_7  |  regular
    pa  |  cpc_2         |  pa_{train,dev}_cpc_2_tensor_400.scp |  (257 by 400)  |  pa_{train,dev}_utt2index_7  |  regular
    la  |  spec (slide)  |  la_{train,dev}_spec_tensor4.scp     |  (257 by 400)  |  la_{train,dev}_utt2index_8  |  regular
    pa  |  spec (slide)  |  pa_{train,dev}_spec_tensor4.scp     |  (257 by 400)  |  pa_{train,dev}_utt2index_8  |  regular
    """
