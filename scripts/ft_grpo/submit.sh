#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=12:00:00

# model tag
tag=$1
# checkpoint of the post-trained SSL
ini_ssl=$2
# csv for training and validation data
trncsv=$3
valcsv=$4
# folder for evaluation csv
prodir=$5
# base
basedir=$6
# data
datadir=$7
# whether skip the training
skip_train=$8

# training
if [[ -z "${skip_train}" ]]
then
    bash script_trn.sh ${tag} ${trncsv} ${valcsv} $PWD/hparams/${tag}.yaml ${prjdir} ${ini_ssl} ${basedir} ${datadir}
else
    echo "skip training for ${tag}"
fi

# inference
# deepfake eval
for evalset in DF_Eval_2024_4 DF_Eval_2024_10  DF_Eval_2024_13 DF_Eval_2024_30 DF_Eval_2024_50
do
    bash script_inf.sh ${tag} ${prodir}/${subdir}/${evalset}.csv ${evalset}.csv $PWD/hparams/${tag}.yaml ${prjdir} ${ini_ssl} ${basedir} ${datadir}
done

# others
# asvspoof2019-la-eval ASVspoof5_eval
for evalset in ADD2023 FakeOrReal_test DeepVoice In-the-wild
do
    bash script_inf.sh ${tag} ${prodir}/${evalset}.csv ${evalset}.csv $PWD/hparams/${tag}.yaml ${prjdir} ${ini_ssl} ${basedir} ${datadir}
done
