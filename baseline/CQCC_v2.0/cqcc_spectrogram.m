function  [CQcepstrum] = ...
    cqcc_spectrogram(x, fs, B, fmax, fmin, d, preE, cmvn)

%%% DEFAULT INPUT PARAMETERS
if nargin < 3; B = 96; end
if nargin < 4; fmax = fs/2; end
if nargin < 5; oct = ceil(log2(fmax/20)); fmin = fmax/2^oct; end
if nargin < 6; d = 16; end
if nargin < 7; cf = 19; end
if nargin < 8; ZsdD = 'ZsdD'; end
if nargin < 9; f_d = 2; end
if nargin < 10; vad = 0; end
if nargin < 11; preE = 0; end
if nargin < 12; arte = 0; end
if nargin < 13; yw_cf = 3; end
if nargin < 14; cmvn = 0; end
gamma = 228.7*(2^(1/B)-2^(-1/B));

%%% PRE-EMPHASIS
x_original = x;
if preE
    x = filter( [1 -0.97], 1, x);
end

%%% CQT COMPUTING
Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma);

%%% LOG POWER SPECTRUM
absCQT = abs(Xcq.c);
dt = Xcq.xlen/size(absCQT,2)/fs;
TimeVec = (1:size(absCQT,2))*dt;
FreqVec = fmin*(2.^((0:size(absCQT,1)-1)/B));
LogP_absCQT = log(absCQT.^2 + eps);

%%% UNIFORM RESAMPLING
%kl = (B*log2(1+1/d));
%[Ures_LogP_absCQT, Ures_FreqVec] = resample(LogP_absCQT,...
%    FreqVec,1/(fmin*(2^(kl/B)-1)),1,1,'spline');

%%% DCT
%CQcepstrum = dct(Ures_LogP_absCQT);
CQcepstrum = LogP_absCQT;

%%% CMVN
%if cmvn
%    CQcc_mu = mean(CQcepstrum,2);
%    CQcc_std = std(CQcepstrum, [], 2);
%    CQcepstrum = bsxfun(@minus, CQcepstrum, CQcc_mu);
%    CQcepstrum = bsxfun(@rdivide, CQcepstrum, CQcc_std);
%end

end
