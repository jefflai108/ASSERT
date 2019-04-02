clear; close all; clc;

% add required libraries to the path
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('CQCC_v2.0'));
addpath(genpath('GMM'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('tDCF_v1'));

% set here the experiment to run (access and feature type)
access_type = 'LA'; % LA for logical or PA for physical
feature_type = 'CQCC'; % LFCC or CQCC

% set paths to the wave files and protocols

% TODO: in this code we assume that the data follows the directory structure:
%
% ASVspoof_root/
%   |- LA
%      |- ASVspoof2019_LA_dev_asv_scores_v1.txt
% 	   |- ASVspoof2019_LA_dev_v1/
% 	   |- ASVspoof2019_LA_protocols_v1/
% 	   |- ASVspoof2019_LA_train_v1/
%   |- PA
%      |- ASVspoof2019_PA_dev_asv_scores_v1.txt
%      |- ASVspoof2019_PA_dev_v1/
%      |- ASVspoof2019_PA_protocols_v1/
%      |- ASVspoof2019_PA_train_v1/

pathToASVspoof2019Data = '/export/b14/jlai/ASVspoof2019-data/';

evaluate_tDCF_asvspoof19(fullfile('cm_scores', ['scores_cm_baseline_' access_type '_' feature_type '.txt']), ...
    fullfile(pathToASVspoof2019Data, access_type, ['ASVspoof2019_' access_type '_dev_asv_scores_v1.txt']));

fullfile('cm_scores', ['scores_cm_baseline_' access_type '_' feature_type '.txt'])
fullfile(pathToASVspoof2019Data, access_type, ['ASVspoof2019_' access_type '_dev_asv_scores_v1.txt'])
