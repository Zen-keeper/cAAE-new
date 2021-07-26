# cAAE

code for Unsupervised Detection of Lesions in Brain MRI using constrained adversarial auto-encoders, [https://arxiv.org/abs/1806.04972](https://arxiv.org/abs/1806.04972)

AAE model is implemented based on [this github repo](https://github.com/Naresh1318/Adversarial_Autoencoder)

Required: python>=2.7, tensorlayer, tensorflow>=1.0, numpy

train model:  python main.py --model_name "cAAE" --z_dim "128" 

## TRAIN

For our project, we can use main_ori_lamda1 as the entrance file (there maybe a little difference with final version) and we fix the param z_dim 128.

## data
Our data is too large to be stored so that we have not provide it.

Please forgive me for the confusion of the current code In the project, and please make sure the data of the last input model through debugging

