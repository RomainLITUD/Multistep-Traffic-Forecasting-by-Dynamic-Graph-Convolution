# Multistep Traffic Forecasting by Dynamic Graph Convolution: Interpretations of Real-Time Spatial Correlations

## Online demo

We have an online Streamlit App to show how the extended version of this model works. Please go to [this page](https://nltrafficpredictiondemo-ekpqd7e8sdue92ewyttuuw.streamlit.app/) for the interactive demo. The related code is available [here](https://github.com/RomainLITUD/NL_Traffic_Prediction_Demo).

## !UPDATE!
This repository uses a very old version of Tensorflow (1.14.0) and Keras (2.3.1). To make the code easier to use, we extend the dynamic graph convolutional module to uncertainty-aware multiple traffic quantities prediction and demand estimation. The new source code converts the layers in this paper to the newest version of PyTorch. An interactive online web application is also provided to give predictions and model interpretations. Please go to [this new page](https://github.com/RomainLITUD/uncertainty-aware-traffic-speed-flow-demand-prediction) for more details. This old repository will not be updated.

-------------------------------------------------
-------------------------------------------------
-------------------------------------------------

## Old description

This is the open source Keras code of the proposed Dynamic Graph Convolutional Networks (DGCN), a multistep network-level traffic condition forecasting model that can capture and explicitly give understandable spatial correlations among road links.

An example of the dataset used in the article (RotCC2) can be downloaded here: https://drive.google.com/file/d/1UCWmA-vLp3LSu1IFSiwdVMXSvdfsVFf9/view?usp=sharing. 

For more datasets please visit DittLab-TUD: https://dittlab.tudelft.nl/, or our online traffic dynamics visualization website: http://dittlab-apps.tudelft.nl/apps/app-ndw/home.jsp, or directly send an email to one of the author:  G.Li-5@tudelft.nl

The meta-description of the dataset is as follows. `x_train` is the observed speed, `e_train` is the input labels for scheduled sampling, `y_train` is labeld to be predicted, the same for test set:
```bash
x_train = Data['Speed_obs_train']
y_train = Data['Speed_pred_train']
e_train = Data['E_train']
x_test = Data['Speed_obs_test']
y_test = Data['Speed_pred_test']
e_test = Data['E_test']
```

To reproduce the results in the paper, please put the corresponding datasets in the "Datasets" file. A command-line parsed `.py` file will be added before 1st March.

![archi](https://user-images.githubusercontent.com/48381256/98677777-cd67e180-235d-11eb-9fd6-4aaaefc790f1.PNG)


## File strucuture
```bash
.
|-custom_models
  |-layers_keras.py                # custom keras layers and DGCN RNN cell
  |-model_keras.py                 # construct DGCN model
  |-math_utils.py                  # mathematical tools
|-pretrained                  # pre-trained models to reproduce the results in the paper
|-DGCRNN.ipynb                # train/test the model, visualize predictions
|-model_interpretation.ipynb  # interpret dynamic spatial correlations
|-utils_vis.py                # visulization tools
```

## Requirements
* scipy 0.19.0
* numpy 1.12.1
* h5py
* statsmodels
* tensorflow 1.14.0 or 1.15.0
* keras 2.3.1 or 2.2.5
* networkx 2.5.0 (for tracking attention distribution in a complex graph)

\* for early versions of tensorflow and keras the modelcheckpoint may fail.

