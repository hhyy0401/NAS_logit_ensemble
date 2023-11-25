# NAS_logit_ensemble

The code is for predicting neural architectures (especially for NAS-Bench-201 dataset).
This code is composed of two steps:
1. Pre-training step: Pre-train logits of top-k (in terms of entropy) images, respectively. 
2. Fine-tuning: Ensemble all predictors to estimate performance of neural architectures.

## Running Codes
- Command "python runner.py setting.json {device}".
  - If CUDA available, set *device=cuda*. Else, set *device=cpu*.
  - You might use "CUDA_VISIBLE_DEVICES={num}" if cuda availble. For example, "CUDA_VISIBLE_DEVICES=0 python runner.py setting.json cuda"
  - You can select hyperparameters in *setting.json* file.
    1. *method=logits* for pre-training logits. *method=gnn* for baseline (without pre-training).
    2. *fig=the number of images*

## Outputs
- Evaluation metrics: mse, mae, R2, spearmanr coefficient, kendall tau, etc. 

## Datasets
- Logit files are on the [link](https://postechackr-my.sharepoint.com/:f:/g/personal/hhyy0401_postech_ac_kr/Ekcx1Ah3JrhOmYYArZXPcF4B9JsU2Sla86-o8gBGvdY5Iw?e=aC5kBh).
- Download files above and save them in the *data>logits* directory.
  - *logits.pt* is sorted in terms of entropy, containing 30 images.
  - *random_logits.pt* consist of logits of randomly selected 30 images.
- *labels.pt* is true labels of NAS-Bench-201 datasets.
- *permutation.npy* contains 50 cases of permutations of NAS-Bench-201 datasets.
- *nas201_sample.pt* contains non-isomorphic 6466 architectures with their information of NAS-Bench-201.
