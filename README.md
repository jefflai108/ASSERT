# ASSERT
Johns Hopkins University's system submission for ASVspoof 2019: ranked 3rd (3/50) in PA sub-challenge and 14th (14/52) in LA sub-challenge. Submitted to Interspeech 2019. 
 
This repository contains codes to reproduce the core results from our paper: 
* 

<p align="center">
 <img src="img/network.png" width="40%">
</p>

## Abstract 
We present JHU's system submission to the ASVspoof 2019 Challenge: Anti-Spoofing with Squeeze-Excitation and Residual neTworks (ASSERT). Anti-spoofing has gathered more and more attention since the inauguration of the ASVspoof Challenges, and ASVspoof 2019 dedicates to address attacks from all three major types: text-to-speech, voice conversion, and replay. Built upon previous research work on Deep Neural Network (DNN), ASSERT is a pipeline for DNN-based approach to anti-spoofing. ASSERT has four components: feature engineering, DNN models, network optimization and system combination, where the DNN models are variants of squeeze-excitation and residual networks. We conducted an ablation study of the effectiveness of each component on the ASVspoof 2019 corpus, and experimental results showed that ASSERT obtained more than 93% and 17% relative improvements over the baseline systems in the two sub-challenges in ASVspooof 2019, ranking ASSERT one of the top performing systems. 

## Dependencies
This project uses Python 2.7. Before running the code, you have to install
* [PyTorch 0.4](https://pytorch.org)
* [Sacred](https://sacred.readthedocs.io/en/latest/index.html)
* [Kaldi](https://github.com/kaldi-asr/kaldi)

The former 2 dependencies can be installed using pip by running
```
pip install -r requirements.txt
```

## Getting Started 

## Authors 
Cheng-I Lai, [Nanxin Chen](http://myemacs.com), [Jesús Villalba](https://www.clsp.jhu.edu/faculty/jesus-villalba/), [Najim Dehak](https://engineering.jhu.edu/ece/faculty/najim-dehak/)

If you encouter any problem, feel free to contact me.
