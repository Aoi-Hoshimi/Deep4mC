# Deep4mC
Using EL & DL methods to predict DNA 4mC sites.


# Updates
2024/12/7:
Proposed a method called feature_selection_with_lazypredict, which can evaluate different feature encoding methods,
based on 3 ML methods testing on 3 species dataset.

2024/12/11：
Created a preliminary CNN PreTraining model and a CatBoost model. Conducted further evaluation and training.

2024/12/17
Created a Transformer model.
Pretrained CNN and Transformer model.(Pre-trained models have beem saved at 'prepare/pretrained_models') 
Created an ensemble model based on stacking.
Uploaded the figure of the overall structure of proposed model.

2024/12/24
Deployed the project to a Linux environment.
Changed the tensorflow environment to tensorflow-gpu environment
Created a B-LSTM model & Stacking-CNN-BLSTM models.
Pretrained CNN, Transformer model and Stacking models.


# Environment requirements
python=3.8, tensorflow-gpu=2.5.0, cuda≥11.2

conda environments can be loaded by:

conda env create -f environment.yml