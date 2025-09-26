<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/logo.png?raw=true" width="60%" alt="AntiDeepfake" />
</div>

## Introduction to Drift Detection

This is a side project to monitor the data drift in the MLOps context of deepfake detection -- we wish to compute a drift value indicating how far away the incoming data is away from a reference data.



<div align="center">
<img src="https://arxiv.org/html/2509.10086v1/x1.png" width="60%" alt="AntiDeepfake" />
</div>

This is detailed in paper
```
Towards Data Drift Monitoring for Speech Deepfake Detection in the context of MLOps
Xin Wang, Wanying Ge, Junichi Yamagishi
https://arxiv.org/abs/2509.10086
```
## Implementation

What we need to compute the drift value:
1. Feature embeddings from each utterance in the incoming dataset
2. Feature embeddings from each utterance in a reference dataset
3. Ways to estimate the distribution of  the feature embeddings
4. Functions to measure the distances (i.e., drift values) between the distributions

### Embedding extraction

AntiDeepfake code supports feature embeddings (the 1st and 2nd points above). 

* To extract the score (csv file) and the utterance-level embedding vectors (pkl file), please use the following command during inference:
```bash
# inference in analysus mode
python main.py analysis hparams/<model>.yaml --base_path <basepath> --output_folder <outputfolder> --pretrained_weights '{"detector": "<pretrained>"}' --test_csv <test_protocol> --score_path <outputfolder>/<output_file_name>.csv

# where
# <model>: model name
# <basepath>: path to the AntiDeepfake (see README.md)
# <outputfolder>: folder to save the dumped score and embedding file
# <test_protocol>: the data protocol for inference
# <output_file_name>: name of the output file
```

* Output should be saved as
    * `<outputfolder>/<output_file_name>.csv`: a CSV file contains the scores per utterance (same format as the default AntiDeepfake format)
    * `<outputfolder>/<output_file_name>.pkl`: a pickle file contains the embedding per utterance. It is a np.array of shape `(N, dims)`, where `N` is the number of uttereances and `dims` is the number of feature dimensions.
    * See notebook below for APIs to load and parse the data

### Drift computation

Functions to comptue the drift values are included in this Jupyter notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nJvR2IDMM9JjSv4YYRbhudm-m7LmRDrc?usp=sharing)


