#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
# script_trn.sh hparam_tag train_csv valid_csv yaml prjdir pretrained

# name of the environment
NAME_ENV=ultra-ssl

tag=$1
yaml=$4
outputfolder=$PWD/exp_${tag}
pretrained=$6

if [ ! -f ${pretrained} ]; then
    echo "Cannot find ${pretrained}"
else
    source ~/.bashrc
    conda activate ${NAME_ENV}

    echo "Submitted job ID: $JOB_ID"

    cd $5

    command="python main.py ${yaml}
		--base_path $7
		--data_folder $8
		--output_folder ${outputfolder}
		--pretrained_weights '{\"detector\": \"${pretrained}\"}'
		--train_csv $2
		--valid_csv $3"
    echo ${command}
    eval ${command}
fi
