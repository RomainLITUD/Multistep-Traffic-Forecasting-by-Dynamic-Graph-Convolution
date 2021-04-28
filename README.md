# Multistep Traffic Forecasting by Dynamic Graph Convolution: Interpretations of Real-Time Spatial Correlations

This is the open source Keras code of the proposed Dynamic Graph Convolutional Networks (DGCN), a multistep network-level traffic condition forecasting model that can capture and explicitly give explainable spatial correlations among road links. The code will be continously optimized and updated later.

An example of the dataset used in the article (RotCC2) can be downloaded here: https://drive.google.com/file/d/1UCWmA-vLp3LSu1IFSiwdVMXSvdfsVFf9/view?usp=sharing.

For more datasets please visit DittLab-TUD: https://dittlab.tudelft.nl/,

or our online traffic dynamics visualization website: http://dittlab-apps.tudelft.nl/apps/app-ndw/home.jsp,

or directly send an email to one of the author:  G.Li-5@tudelft.nl

To reproduce the results in the paper, please put the corresponding datasets in the "Datasets" file

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

