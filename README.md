# NAS_logit_ensemble

The code is for predicting neural architectures (especially for NAS-Bench-201 dataset).
This code is composed of two steps:
1. Pretraining step: Pre-train logits of top-k (in terms of entropy) images, respectively. 
2. Fine-tuning: Ensemble all predictors to estimate performance of neural architectures.

## Running Codes
- Command "python runner.py setting.json {device}".
  - If CUDA available, set *device=cuda*. Else, set *device=cpu*.
  - You might use "CUDA_VISIBLE_DEVICES={num}" if cuda availble.
  - You can select hyperparameters in *setting.json* file.
    <Examples>
    1. *method=logits* for pre-training logits. *method=gnn* for baseline (without pre-training)
    2. *fig=the number of images*

## Outputs
- Evaluation measures including mse, mae, R2, spearmanr coefficient, and kendall tau. 

## Datasets
- Logit files are on the link below.
- You might contain folder name *logits* in the data folder.
- *logits.pt* is sorted in terms of entropy, containing 30 images.
- *random_logits.pt* is randomly selected 30 images.
- https://postechackr-my.sharepoint.com/:f:/g/personal/hhyy0401_postech_ac_kr/Ekcx1Ah3JrhOmYYArZXPcF4B9JsU2Sla86-o8gBGvdY5Iw?e=aC5kBh
