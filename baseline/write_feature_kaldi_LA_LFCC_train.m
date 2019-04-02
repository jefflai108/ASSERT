clear; close all; clc;

% add required libraries to the path
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('kaldi-to-matlab'));

% set here the experiment to run (access and feature type)
access_type = 'LA'; % LA for logical or PA for physical
feature_type = 'LFCC'; % LFCC or CQCC
data_type = 'train'; % train or dev

pathToASVspoof2019Data = '/export/b14/jlai/ASVspoof2019-data/';

pathToDatabase = fullfile(pathToASVspoof2019Data, access_type);
if strcmp(data_type, 'train') % train 
    ProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.train.trn.txt'));
elseif strcmp(data_type, 'dev') % dev
    ProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.dev.trl.txt'));
end

% read protocol
fileID = fopen(ProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file lists
filelist = protocol{2};

%% Feature extraction for data
disp('Extracting features for all data...');
allFeatureCell = cell(size(filelist));
allUttCell = cell(size(filelist));
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,['ASVspoof2019_' access_type strcat('_', data_type, '/flac')],[filelist{i} '.flac']);
    [x,fs] = audioread(filePath);
    if strcmp(feature_type,'LFCC')
        [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    elseif strcmp(feature_type,'CQCC')
        allFeatureCell{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'Z');
    end
    allUttCell{i} = filelist{i};
end
disp('Done!');

%% writekaldifeatures
features = struct;
features.utt = allUttCell;
features.feature = allFeatureCell; 
writekaldifeatures(features, strcat('feats/', access_type, '_', feature_type, '_', data_type, '.ark'));
disp('finish stage 3')
