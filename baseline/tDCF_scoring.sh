#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
tDCF_scoring
EOF
