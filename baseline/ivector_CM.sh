#!/bin/bash

matlab -nodisplay -nosplash <<EOF
maxNumCompThreads(1);
ivector_CM
EOF
