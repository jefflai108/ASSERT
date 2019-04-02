#!/bin/bash 
. ./path.sh

#for name in LA_CQCC_dev LA_CQCC_train LA_LFCC_dev LA_LFCC_train PA_CQCC_dev PA_CQCC_train PA_LFCC_dev PA_LFCC_train; do 
for name in PA_CQCC_eval; do
	copy-feats ark,t:feats/${name}.txt ark,scp:`pwd`/feats/${name}.ark,`pwd`/feats/${name}.scp
done
