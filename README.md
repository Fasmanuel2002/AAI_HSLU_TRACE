### A PyTorch re-implementation of the [TRACE](https://arxiv.org/abs/2409.12972) paper, developed in collaboration with the university.
 
 <p align="center">
 <img width="700" height="627" alt="image" src="https://github.com/user-attachments/assets/340938a8-5ae1-44e3-824c-cf1d6e68f679" />
 </p>

This research project re-implements the **Transformer-based User Representation from Attributed Clickstream Event Sequence (TRACE)** architecture using alternative datasets, as the original dataset provided in the paper is not publicly available. TRACE proposes a novel approach for learning rich user representations from live, multi-session clickstream data with sparse targets.

The TRACE architecture is built around a multi-task learning (MTL) framework, in which a shared Transformer encoder is trained to predict multiple user engagement objectives from sequences of clickstream events. By predicting a diverse set of future user engagement signals, the architecture is encouraged to learn robust and versatile user representations that generalize across tasks and sessions.

<p align="center">
  <img width="682" alt="Training Pipeline" src="https://github.com/user-attachments/assets/4c82b31d-24c1-457d-8569-6bf3ca004317" />
</p>


## Installation Dependencies
The first step to run this project locally or on GPUHub is to install the required dependencies.
```bash
pip install -r requirements.txt
```
Dependencies:

 - [NumPy](https://numpy.org/doc/stable/index.html): fundamental package for scientific computing in Python
 
 - [Pandas](https://pandas.pydata.org/): fundamental package for source data analysis and manipulation tool
 
 - [Pytorch](https://pytorch.org/): software-based open source deep learning framework used to build neural networks
 
 - [Scikit-learn](https://scikit-learn.org/stable/): open-source machine learning library for Python

 - [Seaborn](https://seaborn.pydata.org/): data visualization library based on matplotlib

 - [matplotlib](https://matplotlib.org/): Comprehensive library for creating static, animated, and interactive visualizations in Python
 
 - [tensorboard](https://www.tensorflow.org/tensorboard): Part of the TensorFlow Framework that provides tracking and visualizing of the metrics 


## Installation Dataset 
The second step consists of downloading the dataset used in this investigation from the [OTTO RecSys Dataset](https://github.com/otto-de/recsys-dataset).

You need to visit the following website to download the dataset:
[Download from Kaggle](https://www.kaggle.com/competitions/otto-recommender-system/data)

Download the `train.jsonl` file, which contains approximately **11 GB** of training data.

## Repository Structure

- `Otto_TRACE/dataset/`  
  Contains the dataset processing pipeline based on the OTTO dataset. This class loads data from `train.jsonl`, constructs the model inputs, generates task-specific logits, and performs the input–target split used for training and evaluation.
  
- `Otto_TRACE/model/`
  
  This section presents the re-implementation of the TRACE model architecture, as described in Section 2.3 (Model Architecture) of the original TRACE paper, adapted for this research investigation
  
- `Otto_TRACE/training_models/`
  
  Contains the two versions of training pipeline for Singular-Task Learning (STL) and Multi-Task Learning (MLT), the purpose of this code format is for jupyterhub GPU.

- `Otto_TRACE/test_models/`
  
  Contains the two versions of Testing Pipeline for Singular-Task Learning (STL) and Multi-Task Learning (MLT).

- `Otto_TRACE/utils/`
  
  Contains six files for corresponding reasons
  - EarlyStopping: Stop training when a monitored metric F1 has stopped improving
  - feature_engineering: This file presents the re-implementation of the TRACE feature engineering, as described in Section 2.2 (Feature and Position Encoding) of the original TRACE paper
  - normalization: This file presents the re-implementation of TRACE for normalizating and log the time elapsed and time betwen from Section 2.2 (Feature and Position Encoding) of the original TRACE paper
  - plot_confussion_matrix: Script for Computing and Plotting the Confusion Matrix
  - SplitData: Script for splitting the dataset into training, validation, and test sets.
  - training_utils: Utility scripts designed to improve code readability, modularity, and maintainability.
