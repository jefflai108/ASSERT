#!/bin/bash
# extract fbank, mfcc, logspec, ivector features for:
# ASVspoof2019 LA train, LA dev, LA eval, PA train, PA dev, PA eval

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
fbankdir=`pwd`/fbank
specdir=`pwd`/logspec
vadir=`pwd`/mfcc

stage=1

if [ $stage -eq 0 ]; then
	# first create spk2utt 
	for name in la_eval pa_eval; do
		utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
		utils/fix_data_dir.sh data/${name}
	
	# feature extraction
    	# logspec (257)
	# create a copy of data/la_eval, pa_eval, for logspec

		# logspec
		utils/copy_data_dir.sh data/${name} data/${name}_spec
		local/make_spectrogram.sh --fbank-config conf/spec.conf --nj 40 --cmd "$train_cmd" \
		  data/${name}_spec exp/make_spec $specdir
		utils/fix_data_dir.sh  data/${name}_spec
	
	# apply cm for the extracted features 
	# cm is 3-second sliding window
		
		# logspec
		utils/copy_data_dir.sh data/${name}_spec data/${name}_spec_cm
		feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_spec/feats.scp ark:- |"
		copy-feats "$feats" ark,scp:`pwd`/data/${name}_spec_cm/feats.ark,`pwd`/data/${name}_spec_cm/feats.scp
		utils/fix_data_dir.sh  data/${name}_spec_cm
      done 
fi 


if [ $stage -eq 1 ]; then 
	# prepare cqcc and lfcc data directories
	for name in la_eval pa_eval; do
		utils/copy_data_dir.sh data/${name} data/${name}_cqcc
		#utils/copy_data_dir.sh data/${name} data/${name}_lfcc
	done
	# copy feats.scp 
	for condition in la pa; do 	
	for feats in cqcc; do 
	for mode in eval; do
		cp /export/c03/jlai/ASVspoof2019/baseline/feats/${condition^^}_${feats^^}_${mode}_fixed.scp data/${condition}_${mode}_${feats}/feats.scp
	done	
	done
	done
	
	# apply cm for cqcc and lfcc
	# cm is 3-second sliding window
	for name in la_eval pa_eval; do
		# fix feats.scp sorting 
		utils/fix_data_dir.sh data/${name}_cqcc
		#utils/fix_data_dir.sh data/${name}_lfcc

		utils/copy_data_dir.sh data/${name}_cqcc data/${name}_cqcc_cm
		feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_cqcc/feats.scp ark:- |"
		copy-feats "$feats" ark,scp:`pwd`/data/${name}_cqcc_cm/feats.ark,`pwd`/data/${name}_cqcc_cm/feats.scp
		utils/fix_data_dir.sh  data/${name}_cqcc_cm
		
		#utils/copy_data_dir.sh data/${name}_lfcc data/${name}_lfcc_cm
		#feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_lfcc/feats.scp ark:- |"
		#copy-feats "$feats" ark,scp:`pwd`/data/${name}_lfcc_cm/feats.ark,`pwd`/data/${name}_lfcc_cm/feats.scp
		#utils/fix_data_dir.sh  data/${name}_lfcc_cm
	done
	
	# compute vad based on mfcc
	#for name in la_train la_dev pa_train pa_dev; do
	#	sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
	#		data/${name}_mfcc exp/make_vad $vadir
	#done
	
	# copy vad.scp to each data directory
	#for name in la_train la_dev pa_train pa_dev; do
	#	for feats in fbank fbank_cm spec spec_cm mfcc_cm; do
	#		cp data/${name}_mfcc/vad.scp data/${name}_${feats}
	#	done
	#done
fi 


if [ $stage -eq 100 ]; then
	# first create spk2utt 
	for name in la_train la_dev pa_train pa_dev; do
		utils/utt2spk_to_spk2utt.pl data/${name}/utt2spk > data/${name}/spk2utt
		utils/fix_data_dir.sh data/${name}
	
	# feature extraction
    	# mfcc (24), fbank (40), logspec (257)
	# create a copy of data/la_train, la_dev, pa_train, pa_dev for mfcc, fbank & logspec

		# mfcc
		utils/copy_data_dir.sh data/${name} data/${name}_mfcc
		steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
			data/${name}_mfcc exp/make_mfcc $mfccdir
		utils/fix_data_dir.sh  data/${name}_mfcc
		# fbank
		utils/copy_data_dir.sh data/${name} data/${name}_fbank
		steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
			data/${name}_fbank exp/make_fbank $fbankdir     	
		utils/fix_data_dir.sh  data/${name}_fbank
		# logspec
		utils/copy_data_dir.sh data/${name} data/${name}_spec
		local/make_spectrogram.sh --fbank-config conf/spec.conf --nj 40 --cmd "$train_cmd" \
		  data/${name}_spec exp/make_spec $specdir
		utils/fix_data_dir.sh  data/${name}_spec
	
	# apply cm for the extracted features 
	# cm is 3-second sliding window
		
		# mfcc
		utils/copy_data_dir.sh data/${name}_mfcc data/${name}_mfcc_cm
		feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_mfcc/feats.scp ark:- |"
		copy-feats "$feats" ark,scp:`pwd`/data/${name}_mfcc_cm/feats.ark,`pwd`/data/${name}_mfcc_cm/feats.scp
		utils/fix_data_dir.sh  data/${name}_mfcc_cm
		
		# fbank
		utils/copy_data_dir.sh data/${name}_fbank data/${name}_fbank_cm
		feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_fbank/feats.scp ark:- |"
		copy-feats "$feats" ark,scp:`pwd`/data/${name}_fbank_cm/feats.ark,`pwd`/data/${name}_fbank_cm/feats.scp
		utils/fix_data_dir.sh  data/${name}_fbank_cm
	
		# logspec
		utils/copy_data_dir.sh data/${name}_spec data/${name}_spec_cm
		feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_spec/feats.scp ark:- |"
		copy-feats "$feats" ark,scp:`pwd`/data/${name}_spec_cm/feats.ark,`pwd`/data/${name}_spec_cm/feats.scp
		utils/fix_data_dir.sh  data/${name}_spec_cm
      done 
fi 


if [ $stage -eq 150 ]; then 
	# prepare cqcc and lfcc data directories
	for name in la_train la_dev pa_train pa_dev; do
		utils/copy_data_dir.sh data/${name} data/${name}_cqcc
		#utils/copy_data_dir.sh data/${name} data/${name}_lfcc
	done
	# copy feats.scp 
	for condition in la pa; do 	
	for feats in cqcc; do 
	for mode in train dev; do
		cp /export/c03/jlai/ASVspoof2019/baseline/feats/${condition^^}_${feats^^}_${mode}_fixed.scp data/${condition}_${mode}_${feats}/feats.scp
	done	
	done
	done
	
	# apply cm for cqcc and lfcc
	# cm is 3-second sliding window
	for name in la_train la_dev pa_train pa_dev; do
		# fix feats.scp sorting 
		utils/fix_data_dir.sh data/${name}_cqcc
		#utils/fix_data_dir.sh data/${name}_lfcc

		utils/copy_data_dir.sh data/${name}_cqcc data/${name}_cqcc_cm
		feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_cqcc/feats.scp ark:- |"
		copy-feats "$feats" ark,scp:`pwd`/data/${name}_cqcc_cm/feats.ark,`pwd`/data/${name}_cqcc_cm/feats.scp
		utils/fix_data_dir.sh  data/${name}_cqcc_cm
		
		#utils/copy_data_dir.sh data/${name}_lfcc data/${name}_lfcc_cm
		#feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:`pwd`/data/${name}_lfcc/feats.scp ark:- |"
		#copy-feats "$feats" ark,scp:`pwd`/data/${name}_lfcc_cm/feats.ark,`pwd`/data/${name}_lfcc_cm/feats.scp
		#utils/fix_data_dir.sh  data/${name}_lfcc_cm
	done
	
	# compute vad based on mfcc
	#for name in la_train la_dev pa_train pa_dev; do
	#	sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
	#		data/${name}_mfcc exp/make_vad $vadir
	#done
	
	# copy vad.scp to each data directory
	#for name in la_train la_dev pa_train pa_dev; do
	#	for feats in fbank fbank_cm spec spec_cm mfcc_cm; do
	#		cp data/${name}_mfcc/vad.scp data/${name}_${feats}
	#	done
	#done
fi 


if [ $stage -eq 199 ]; then
	
	num_components=64 # UBM
	ivector_dim=100 # ivector
	
	# i-vector extractor training 
	for name in la_train pa_train; do
		for feats in cqcc mfcc; do 
			# ivector extractor with full-covariance UBM trained with 
			# data/la_train_cqcc
			# data/pa_train_cqcc
			local/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
				--nj 20 --num-threads 8 --subsample 1 \
				data/${name}_${feats} $num_components \
				exp/${num_components}_diag_ubm_${name}_${feats}

			local/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
				--nj 20 --remove-low-count-gaussians false --subsample 1 \
				data/${name}_${feats} \
				exp/${num_components}_diag_ubm_${name}_${feats} exp/${num_components}_full_ubm_${name}_${feats}

			local/train_ivector_extractor.sh --cmd "$train_cmd --mem 20G" \
				--ivector-dim $ivector_dim \
				--num-iters 5 \
				--num-threads 2 \
				--num-processes 2 \
				exp/${num_components}_full_ubm_${name}_${feats}/final.ubm data/${name}_${feats} \
				exp/${ivector_dim}_extractor_${name}_${feats}
		done 
	done

	# extract i-vectors 
	for name in la pa; do
		for feats in cqcc mfcc; do
			# extract i-vectors for 
			# la_dev with la_train and pa_dev with pa_train
			local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 20 \
				exp/${ivector_dim}_extractor_${name}_train_${feats} data/${name}_dev_${feats} \
				exp/${ivector_dim}_ivectors_${name}_dev_${feats} 
			# extract i-vectors for 
			# la_train with la_train and pa_train with pa_train 
			local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 20 \
				exp/${ivector_dim}_extractor_${name}_train_${feats} data/${name}_train_${feats} \
				exp/${ivector_dim}_ivectors_${name}_train_${feats} 
		done
	done

	# lda+plda backend training 
	for name in la_train pa_train; do
		for feats in mfcc cqcc; do
			# Compute the mean vector for centering the evaluation i-vectors.
			$train_cmd exp/${ivector_dim}_ivectors_${name}_${feats}/log/compute_mean.log \
				ivector-mean scp:exp/${ivector_dim}_ivectors_${name}_${feats}/ivector.scp \
				exp/${ivector_dim}_ivectors_${name}_${feats}/mean.vec || exit 1;
			# This script uses LDA to decrease the dimensionality prior to PLDA.
  			lda_dim=${ivector_dim}
  			$train_cmd exp/${ivector_dim}_ivectors_${name}_${feats}/log/lda.log \
    				ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    				"ark:ivector-subtract-global-mean scp:exp/${ivector_dim}_ivectors_${name}_${feats}/ivector.scp ark:- |" \
    				ark:data/${name}_${feats}/utt2spk exp/${ivector_dim}_ivectors_${name}_${feats}/transform.mat || exit 1;
			# Train the PLDA model.
  			$train_cmd exp/${ivector_dim}_ivectors_${name}_${feats}/log/plda.log \
    				ivector-compute-plda ark:data/${name}_${feats}/spk2utt \
    				"ark:ivector-subtract-global-mean scp:exp/${ivector_dim}_ivectors_${name}_${feats}/ivector.scp ark:- | transform-vec exp/${ivector_dim}_ivectors_${name}_${feats}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    				exp/${ivector_dim}_ivectors_${name}_${feats}/plda || exit 1;
		done
	done
fi


if [ $stage -eq 200 ]; then
	
	num_components=128 # UBM
	ivector_dim=200 # ivector
	
	# i-vector extractor training 
	for name in la_train pa_train; do
		for feats in cqcc mfcc; do 
			# ivector extractor with full-covariance UBM trained with 
			# data/la_train_cqcc
			# data/pa_train_cqcc
			local/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
				--nj 20 --num-threads 8 --subsample 1 \
				data/${name}_${feats} $num_components \
				exp/${num_components}_diag_ubm_${name}_${feats}

			local/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
				--nj 20 --remove-low-count-gaussians false --subsample 1 \
				data/${name}_${feats} \
				exp/${num_components}_diag_ubm_${name}_${feats} exp/${num_components}_full_ubm_${name}_${feats}

			local/train_ivector_extractor.sh --cmd "$train_cmd --mem 20G" \
				--ivector-dim $ivector_dim \
				--num-iters 5 \
				--num-threads 2 \
				--num-processes 2 \
				exp/${num_components}_full_ubm_${name}_${feats}/final.ubm data/${name}_${feats} \
				exp/${ivector_dim}_extractor_${name}_${feats}
		done 
	done

	# extract i-vectors 
	for name in la pa; do
		for feats in cqcc mfcc; do
			# extract i-vectors for 
			# la_dev with la_train and pa_dev with pa_train
			local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 20 \
				exp/${ivector_dim}_extractor_${name}_train_${feats} data/${name}_dev_${feats} \
				exp/${ivector_dim}_ivectors_${name}_dev_${feats} 
			# extract i-vectors for 
			# la_train with la_train and pa_train with pa_train 
			local/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 20 \
				exp/${ivector_dim}_extractor_${name}_train_${feats} data/${name}_train_${feats} \
				exp/${ivector_dim}_ivectors_${name}_train_${feats} 
		done
	done

	# lda+plda backend training 
	for name in la_train pa_train; do
		for feats in mfcc cqcc; do
			# Compute the mean vector for centering the evaluation i-vectors.
			$train_cmd exp/${ivector_dim}_ivectors_${name}_${feats}/log/compute_mean.log \
				ivector-mean scp:exp/${ivector_dim}_ivectors_${name}_${feats}/ivector.scp \
				exp/${ivector_dim}_ivectors_${name}_${feats}/mean.vec || exit 1;
			# This script uses LDA to decrease the dimensionality prior to PLDA.
  			lda_dim=${ivector_dim}
  			$train_cmd exp/${ivector_dim}_ivectors_${name}_${feats}/log/lda.log \
    				ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    				"ark:ivector-subtract-global-mean scp:exp/${ivector_dim}_ivectors_${name}_${feats}/ivector.scp ark:- |" \
    				ark:data/${name}_${feats}/utt2spk exp/${ivector_dim}_ivectors_${name}_${feats}/transform.mat || exit 1;
			# Train the PLDA model.
  			$train_cmd exp/${ivector_dim}_ivectors_${name}_${feats}/log/plda.log \
    				ivector-compute-plda ark:data/${name}_${feats}/spk2utt \
    				"ark:ivector-subtract-global-mean scp:exp/${ivector_dim}_ivectors_${name}_${feats}/ivector.scp ark:- | transform-vec exp/${ivector_dim}_ivectors_${name}_${feats}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    				exp/${ivector_dim}_ivectors_${name}_${feats}/plda || exit 1;
		done
	done
fi


if [ $stage -eq -3 ]; then
    # logspec (257) feature extraction
    # apply vad 
    for name in train dev eval train_dev; do
      utils/fix_data_dir.sh data/${name}
      utils/copy_data_dir.sh data/${name} data/${name}_spec
      
      local/make_spectrogram.sh --fbank-config conf/spec.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_spec exp/make_spec $specdir
      sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
	  data/${name}_spec exp/make_vad $vadir
      utils/copy_data_dir.sh data/${name}_spec data/${name}_spec_vad
      feats="ark:select-voiced-frames scp:`pwd`/data/${name}_spec/feats.scp scp:`pwd`/data/${name}_spec/vad.scp ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_spec_vad/feats.ark,`pwd`/data/${name}_spec_vad/feats.scp
    done
fi 


if [ $stage -eq -2 ]; then
    # logspec (257) feature extraction
    # apply vad --> cmvn sliding window 
    for name in train dev eval train_dev; do
      utils/fix_data_dir.sh data/${name}
      utils/copy_data_dir.sh data/${name} data/${name}_spec
      
      local/make_spectrogram.sh --fbank-config conf/spec.conf --nj 40 --cmd "$train_cmd" \
          data/${name}_spec exp/make_spec $specdir
      sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
	  data/${name}_spec exp/make_vad $vadir
      utils/copy_data_dir.sh data/${name}_spec data/${name}_spec_vad_cmvn
      feats="ark:select-voiced-frames scp:`pwd`/data/${name}_spec/feats.scp scp:`pwd`/data/${name}_spec/vad.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
      copy-feats "$feats" ark,scp:`pwd`/data/${name}_spec_vad_cmvn/feats.ark,`pwd`/data/${name}_spec_vad_cmvn/feats.scp
    done
fi 
