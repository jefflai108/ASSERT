function y = artefilt(x,fs,Fea,yw_cf)

%   Articulation rate filtering (ARTE filter)
%   Usage:  y = artefilt(x,fs,X)
%
%   Input parameters:
%         x        : input signal
%         fs       : sampling frequency
%         Fea      : features (nCoeff x nFea)
%         yw_cf    : order of Yule-Walker recursive filter [default 3]
%
%   Output parameters:
%         y        : ARTE filter output (nCoeff x nFea)
%
%   See also:  cqcc, cqt
%
%   References:
%     M. Todisco, H. Delgado, and N. Evans. Articulation Rate Filtering of 
%     CQCC Features for Automatic Speaker Verification. Proceeding of 
%     INTERSPEECH: 17th Annual Conference of the International Speech 
%     Communication Association, 2016.
%
%     C. Sch�rkhuber, A. Klapuri, N. Holighaus, and M. D�fler. A Matlab
%     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy
%     Transforms. Proceedings AES 53rd Conference on Semantic Audio, London,
%     UK, Jan. 2014. http://www.cs.tut.fi/sgn/arg/CQT/
%
%     N. Holighaus, M. D�fler, G. Velasco, and T. Grill. A framework for
%     invertible, real-time constant-q transforms. Audio, Speech, and
%     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) 2016 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Authors: Massimiliano Todisco {todisco [at] eurecom [dot] fr}
%          Hector Delgado {delgado [at] eurecom [dot] fr}
%
% Version: 1.0
% Date: 12.06.16
%
% User are requested to cite the following paper in papers which report 
% results obtained with this software package.	
%
%     M. Todisco, H. Delgado, and N. Evans. Articulation Rate Filtering of 
%     CQCC Features for Automatic Speaker Verification. Proceeding of 
%     INTERSPEECH: 17th Annual Conference of the International Speech 
%     Communication Association, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% CHECK INPUT PARAMETERS
if nargin < 3
    warning('Not enough input arguments.'), return
end

%%% DEFAULT INPUT PARAMETERS
if nargin < 4; yw_cf = 3; end
fmin = 0.5; fmax = 32; B = 96;

%%% FEATURES INPUT SIZE
[~,frame_len] = size(Fea);
ts_frame = length(x)/fs/frame_len;
fs_frame = 1/ts_frame;

%%% INPUT SIGNAL ENVELOPE
[b,a]=butter(2,fmax/(fs/2),'low');
z=filtfilt(b,a,abs(x-mean(x)));

%%% DOWNSAMPLED ENVELOPE
fs_low = 10*fmax;
[P,Q] = rat(fs_low/fs);
[bdc,adc] = butter(1,fmin/(fs_low/2),'high');
x_env = filtfilt(bdc,adc,resample(z,P,Q));

%%% CQT COMPUTING
Xcq = cqt(x_env, B, fs_low, fmin, fmax, 'rasterize', 'full', 'gamma', 0);
absCQT = abs(Xcq.c);
FreqVec = (fmin*(2.^((0:size(absCQT,1)-1)/B)))';

%%% EXPONENTIAL WINDOW
cq_f = mean(absCQT,2);
cq_w = [exp(log(0.01)+(log(0.99)-log(0.01))/B:(log(0.99)-log(0.01))/B:log(0.99))'; ...
    ones(size(absCQT,1)-2*B,1);...
        exp(log(0.99):-(log(0.99)-log(0.01))/B:log(0.01)+(log(0.99)-log(0.01))/B)']; 
cq_f = cq_f.*cq_w;

%%% UNIFORM RESAMPLING
A = [eps; cq_f; eps];
F = [eps; FreqVec; 1/(2*ts_frame)];
[res_absCQT, res_FreqVec] = resample(A,F,B/fmin,1,1,'spline');

%%% YULE-WALKER RECURSIVE FILTER
[b,a] = yulewalk(yw_cf,res_FreqVec./res_FreqVec(end),res_absCQT/max(res_absCQT));

%%% APPLYING YULE-WALKER FILTER TO FEATURES
[bl,al] = butter(1,fmin/(fs_frame/2),'high');
b = conv(b,bl);
a = conv(a,al);
y = filtfilt(b, a, Fea')';

end