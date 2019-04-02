#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
enhanced_baseline_CM_LA_CQCC
enhanced_baseline_CM_PA_CQCC
EOF
