CQCC V2.0 FEATURES

Matlab implementation of constant Q cepstral coefficients successfully used for:
- spoofed speech detection (see CQCC V1.0);
- automatic speaker verification.

Copyright (C) 2016 EURECOM, France.

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/

For further details, refer to the following publication:

Todisco, M., Delgado, H., Evans, N., Articulation Rate Filtering of CQCC Features for Automatic Speaker Verification. Proceeding of INTERSPEECH: 17th Annual Conference of the International Speech Communication Association, September 8-13, 2016, San Francisco, USA.

Todisco, M., Delgado, H., Evans, N., A new feature for automatic speaker verification anti-spoofing: Constant Q cepstral coefficients. ODYSSEY 2016, The Speaker and Language Recognition Workshop, June 21-24, 2016, Bilbao, Spain.

Contents of the package
=======================

- CQT_toolbox_2013
------------------
This folder contains Matlab codes written by the authors of the following paper:

Schörkhuber, C., Klapuri, N. Holighaus, and M. Döfler, "A Matlab Toolbox for Efficient Perfect Reconstruction Time-Frequency Transforms with Log-Frequency Resolution," AES 53rd International Conference on Semantic Audio, London, UK

The toolbox can be downloaded from http://www.cs.tut.fi/sgn/arg/CQT/. This software is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/)

- cqcc.m
--------
Function which computes the Constant Q Cepstral Coefficients (CQCC) from a speech signal

- D18_1000001.wav
-----------------
Speech utterance extracted from the ASVspoof 2015 database. This file is provided as part of the demo. Please refer to the following paper:

Wu, Z., Kinnunen, T., Evans, N., Yamagishi, J. (2015). Automatic Speaker Verification Spoofing and Countermeasures Challenge (ASVspoof 2015) Database, [dataset]. University of Edinburgh. The Centre for Speech Technology Research (CSTR). http://dx.doi.org/10.7488/ds/298

The ASVspoof 2015 database is free for commercial and non-commercial use and is released under a Creative Commons Attribution License (CC-BY) (https://creativecommons.org/licenses/by/4.0/). 

- DEMO.m
--------
This demo script demonstrates the use of the "cqcc.m" function. It loads the provided wave file "D18_1000001.wav" and calls the "cqcc.m" function with a set of pre-defined parameters. Those parameters were used successfully for speaker verification system where CQCCs generally outperform standard MFCCs on the RSR2015 and RedDots datasets. 
It has also been demonstrated that fusion experiments show relative reductions in equal error rates of as much as 60% compared to an MFCC baseline. Further details could be found in the following paper:

Todisco, M., Delgado, H., Evans, N., “Articulation Rate Filtering of CQCC Features for Automatic Speaker Verification,” Proceeding of INTERSPEECH: 17th Annual Conference of the International Speech Communication Association, 2016.

Contact information.
====================

For any query, please contact:

- Massimiliano Todisco (todisco at eurecom.fr)
- Hector Delgado (delgado at eurecom.fr)
- Nicholas Evans (evans at eurecom.fr)

