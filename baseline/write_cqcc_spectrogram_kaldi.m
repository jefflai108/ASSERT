%clear; close all; clc;

data_type ='eval';
output_file = 'cqcc_spectrogram_kaldi_'; 

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v2.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('kaldi-to-matlab'));

% set paths to the wave files and protocols
pathToDatabase = fullfile('..','ASVspoof2017_data','data');
trainProtocolFile = fullfile('..','ASVspoof2017_data','list', strcat(data_type,'.txt'));

% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels   = protocol{2};
speakers = protocol{3};
uttlist  = filelist;
% utt name processing
for i=1:length(filelist)
    file = strsplit(filelist{i},'.');
    spk  = speakers{i};
    mod  = strcat(spk,'-',file{1});
    uttlist{i} = mod;
end
disp('finish stage 1')

%% Feature extraction for training data

% extract features store in cell array
disp('Extracting features for data...');
allFeatureCell = cell(size(filelist));
allUttCell = cell(size(filelist));
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,data_type,filelist{i});
    [x,fs] = audioread(filePath);
    allFeatureCell{i} = cqcc_spectrogram(x, fs, 96, fs/2, fs/2^10, 16, 1, 1);   
    allUttCell{i} = uttlist{i};
end
disp('finish stage 2')

%% writekaldifeatures
features = struct;
features.utt = allUttCell;
features.feature = allFeatureCell; 
writekaldifeatures(features,strcat(output_file,data_type,'.ark'));
disp('finish stage 3')
