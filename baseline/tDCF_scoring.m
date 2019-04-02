%% get feature 
clear; close all; clc;

% 100_ivectors_la_dev_cqcc.txt
access_type = 'PA'; % LA for logical or PA for physical

% add required libraries to the path
addpath(genpath('bosaris_toolkit'));
addpath(genpath('tDCF_v1'));
addpath(genpath('kaldi-to-matlab'));

pathToASVspoof2019Data = '/export/b14/jlai/ASVspoof2019-data/';

% compute performance
evaluate_tDCF_asvspoof19(fullfile('cm_scores', '12-scores.txt'), ...
    fullfile(pathToASVspoof2019Data, access_type, ['ASVspoof2019_' access_type '_dev_asv_scores_v1.txt']));
