# NAS_logit_ensemble

The code is for predicting neural architectures (especially for NAS-Bench-201 dataset).
This code is composed of two steps:

1. 


## Running Codes
- Command "python runner.py setting.json {device}".
  - If CUDA available, set *device=cuda*. Else, set *device=cpu*.
  - You might use "CUDA_VISIBLE_DEVICES={num}" if cuda availble.
  - You can select hyperparameters in *setting.json* file.

## Outputs
- Evaluation measures including mse, mae, R2, spearmanr coefficient, and kendall tau. 

## Datasets
- Logit files are on the link below.
- You might contain folder name *logits* in the data folder.
https://postechackr-my.sharepoint.com/:f:/g/personal/hhyy0401_postech_ac_kr/Ekcx1Ah3JrhOmYYArZXPcF4B9JsU2Sla86-o8gBGvdY5Iw?e=aC5kBh
