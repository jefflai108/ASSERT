# ASSERT
Johns Hopkins University's system submission for ASVspoof 2019: ranked 3rd (3/50) in PA sub-challenge and 14th (14/52) in LA sub-challenge. Submitted to Interspeech 2019.
 
## Abstract 
We present JHU's system submission to the ASVspoof 2019 Challenge: Anti-Spoofing with Squeeze-Excitation and Residual neTworks (ASSERT). Anti-spoofing has gathered more and more attention since the inauguration of the ASVspoof Challenges, and ASVspoof 2019 dedicates to address attacks from all three major types: text-to-speech, voice conversion, and replay. Built upon previous research work on Deep Neural Network (DNN), ASSERT is a pipeline for DNN-based approach to anti-spoofing. ASSERT has four components: feature engineering, DNN models, network optimization and system combination, where the DNN models are variants of squeeze-excitation and residual networks. We conducted an ablation study of the effectiveness of each component on the ASVspoof 2019 corpus, and experimental results showed that ASSERT obtained more than 93% and 17% relative improvements over the baseline systems in the two sub-challenges in ASVspooof 2019, ranking ASSERT one of the top performing systems. 

## Getting Started 
1. Prerequisites: [PyTorch 0.4](https://pytorch.org), [Sacred](https://sacred.readthedocs.io/en/latest/index.html), [Kaldi](https://github.com/kaldi-asr/kaldi)
2. main.py contains detials of the training details and configurations
4. src/attention_neuro/simple_attention_network.py contains the implementation of Attentive Filtering Network.

## Authors 
Cheng-I Lai, Nanxin Chen, Jes\'us Villalba, Najim Dehak

## Contact 
Cheng-I Jeff Lai: jefflai108@gmail.com
