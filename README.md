# Multistep Traffic Forecasting by Dynamic Graph Convolution: Interpretations of Real-Time Spatial Correlations

This is the open source Keras code of the proposed Dynamic Graph Convolutional Networks (DGCN), a multistep network-level traffic condition forecasting model that can capture and explicitly give explainable spatial correlations among road links.

An example of the dataset used in the article (RotCC2) can be downloaded here: https://drive.google.com/file/d/1UCWmA-vLp3LSu1IFSiwdVMXSvdfsVFf9/view?usp=sharing.
For more datasets please visit DittLab-TUD: https://dittlab.tudelft.nl/,
or our online traffic dynamics visualization website: http://dittlab-apps.tudelft.nl/apps/app-ndw/home.jsp.

## File strucuture

.
|---custom_models
  |---layers_keras.py
  |---model_keras.py
  |---math-utils.py
|---pretrained
|---DGCRNN.ipynb
|---model_interpretation.ipynb
|---utils_vis.py
