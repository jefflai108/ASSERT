#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
test_tDCF
EOF
