#!/usr/bin/bash

# path to save the post-trained checkpoint 
checkpoint_path=<PATH>/ssl-weights-tuned
# prjdir/main.py will be called
prjdir=<PATH_TO_ANTIDEEPFAKE_FOLDER>
# protocol folder
prodir=<PATH_TO_PROTOCOLS>
# base folder
basedir=<PATH_TO_BASE>
# data folder
datadir=<PATH_TO_DATA>

# train and validation CSV
trncsv=${prodir}/Deepfake-Eval-2024-TSUBAME/train_segment_random.csv
valcsv=${prodir}/asvspoof2019-la-dev.csv

# eval csv folders.
# DF_*.csv, ADD*.csv in this folder will be used
evalcsvdir=${prodir}

# submit jobs for the following post-trained models
for name in mms_300m w2v_small xlsr_1b xlsr_2b w2v_large mms_1b
do
    checkpoint=${checkpoint_path}/${name}.ckpt
    if [ ! -f $checkpoint ];
    then
	echo "Cannot find $checkpoint"
	exit
    fi

    # submit the job to do inference using pre-post model (no training)
    tag=${name}_pre-post
    qsub -g ${gid} -o log_${tag} -e log_${tag}_err submit.sh ${tag} ${checkpoint} ${trncsv} ${valcsv} ${evalcsvdir} ${basedir} ${datadir} "skiptrain"

    # submit the job for fine-tune pre-post models and do inference for 3 times
    for hparam in ${name}_pre-post-grpo_v1 ${name}_pre-post-grpo_v2 ${name}_pre-post-sft
    do
	for ver in 1 2 3
	do
	    tag=${hparam}_${ver}
	    echo ${tag}
	    qsub -g ${gid} -o log_${tag} -e log_${tag}_err submit.sh ${tag} ${checkpoint} ${trncsv} ${valcsv} ${evalcsvdir} ${basedir} ${datadir} 
	done
    done
done
