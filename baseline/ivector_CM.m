%% get feature 
clear; close all; clc;

% add required libraries to the path
addpath(genpath('bosaris_toolkit'));
addpath(genpath('tDCF_v1'));
addpath(genpath('kaldi-to-matlab'));

%ivector_tDCF(100, 'PA', 'CQCC' , 'cosine')
%ivector_tDCF(100, 'PA', 'MFCC' , 'cosine')
%ivector_tDCF(100, 'LA', 'CQCC' , 'cosine')
%ivector_tDCF(100, 'LA', 'MFCC' , 'cosine')

%ivector_tDCF(200, 'PA', 'CQCC' , 'cosine')
%ivector_tDCF(200, 'PA', 'MFCC' , 'cosine')
%ivector_tDCF(200, 'LA', 'CQCC' , 'cosine')
%ivector_tDCF(200, 'LA', 'MFCC' , 'cosine')

ivector_tDCF(200, 'PA', 'CQCC' , 'guassian')
ivector_tDCF(200, 'PA', 'MFCC' , 'guassian')
ivector_tDCF(200, 'LA', 'CQCC' , 'guassian')
ivector_tDCF(200, 'LA', 'MFCC' , 'guassian')
