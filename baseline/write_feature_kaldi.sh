#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
write_feature_kaldi_LA_CQCC_eval
write_feature_kaldi_PA_CQCC_eval
EOF
