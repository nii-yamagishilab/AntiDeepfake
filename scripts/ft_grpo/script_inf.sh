#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
# script_inf.sh hparam_tag test_csv path_save_csv yaml prjdir pretrained

# name of the environment
NAME_ENV=ultra-ssl

tag=$1
yaml=$4
outputfolder=$PWD/exp_${tag}

pretrained=$6

source ~/.bashrc
conda activate ${NAME_ENV}

echo "Submitted job ID: $JOB_ID"

cd $5

if [[ ! -d "${outputfolder}" ]]
then
    mkdir -p ${outputfolder}
fi

if [[ ! -e "${outputfolder}/$3" ]]
then
    command="python main.py inference ${yaml}
		--base_path $7
		--data_folder $8
		--output_folder ${outputfolder}
		--pretrained_weights '{\"detector\": \"${pretrained}\"}'
		--test_csv $2
		--score_path ${outputfolder}/$3"
    echo ${command}
    eval ${command}
else
    echo "${outputfolder}/$3 exists, skip inference"
fi
