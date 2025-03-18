# <div align="center">Improving Anomaly Detection for Electric Vehicle Charging with Generative Adversarial Networks</div>

## Table of Contents

- [About](#about)
- [Structure](#structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
    - [`ROC_curve_ML_Based_IDS.py`](#roc_curve_ml_based_idspy)
    - [`ROC_curve_RetrainML_Based_IDS.py`](#roc_curve_retrainml_based_idspy)
    - [`T_SNE_distribution_data.py`](#t_sne_distribution_datapy)
   

## About

This repository provides the source code for the paper "Improving Anomaly Detection for Electric Vehicle Charging with Generative Adversarial Networks" which is currently under review.

IDS plays a crucial role in monitoring network traffic and identifying potential threats or malicious activities within a network. By leveraging GANs, a novel approach is introduced to enhance the detection accuracy of traditional IDS systems.

## Structure

```structure
├───configs
├───data
│   ├───acn_caltech
│   ├───acn_jpl
│   └───acn_office
├───docs
├───models
│   ├───encoder
│   ├───generator
│   │   ├───acn_caltech
│   │   │   ├───attack
│   │   │   └───normal
│   │   ├───acn_jpl
│   │   │   ├───attack
│   │   │   └───normal
│   │   └───acn_office
│   │       ├───attack
│   │       └───normal
│   ├───ML_models
│   │   ├───acn_caltech
│   │   ├───acn_jpl
│   │   └───acn_office
│   └───regression_model
│       ├───acn_caltech
│       ├───acn_jpl
│       └───acn_office
├───results
├───src
└───tools
```

This source directories are as follows:

**Executable Files in `tools`:**

- **`ROC_curve_ML_Based_IDS.py`** - Generates ROC curves to compare model performance before and after applying adversarial charging session.
- **`ROC_curve_RetrainML_Based_IDS.py`** - Generates ROC curves to compare model performance before and after retraining with adversarial charging session.
- **`T_SNE_distribution_data.py`** - Visualizes distribution charging sessions using t-SNE. 

**Source Directories:**

- **configs** - Includes all configuration files for processing, data generation, and visualization.
- **data** - Includes all data files required for the implementation of the project.
- **docs** - Includes documents related to algorithms implemented in this project.
- **models** - Includes ML and DL models. 
    - **encoders** - Includes pickle file of encoder object for generating the next charge speed prediction.
    - **generator** - Includes pytorch file of the generator, which is responsible for generating adversarial charging sessions.
    - **ML_models** - Includes pickle files of Machine Learning-Based IDS models, used for detecting anomalies, before and after retraining.
    - **regression_model** - Includes a pickle file of the remaining Machine Learning model, used for predicting the next charge speed, and incorporates submodels in an ensemble method.
- **results** - Includes the output figures generated from the module tools.
- **src** - Includes all helper functions for the above executable files.
- **tools** - Includes all the above executable files.

## Getting Started

### Installation

1. Clone project repo

    ```
    git clone https://github.boschdevcloud.com/SDV-Playground/IDS.git](https://github.boschdevcloud.com/SDV-Playground/GAN_IDS.git
    ```

2. Set up a virtual environment
    
    This project requires Anaconda for execution. If you do not have Anaconda installed, please download it from [here](https://docs.anaconda.com/anaconda/install/)
    
    Create a virtual environment name `lstmgan`

    ``` 
    conda create -n lstmgan python
    ```

    Activating the virtual environment

    ```
    conda activate lstmgan
    ```

    Install Pytorch (with CUDA support) 
    
     ```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
 
3. Install project dependencies

    ```
    cd .\GAN_IDS\  
    pip install -e .
    ```

### Usage

#### `ROC_curve_ML_Based_IDS.py`

To generates ROC curves before and after applying adversarial charging session.

```
python .\tools\ROC_curve_ML_Based_IDS.py acn_office attack -s -d
```

For more information:

```
python .\tools\ROC_curve_ML_Based_IDS.py -h
```

#### `ROC_curve_RetrainML_Based_IDS.py`

To generates ROC curves before and after retraining ML-Based IDS with adversarial charging session.

```
python .\tools\ROC_curve_RetrainML_Based_IDS.py acn_office attack -s -d
```

For more information:

```
python .\tools\ROC_curve_RetrainML_Based_IDS.py -h
```

#### `T_SNE_distribution_data.py`

To visualize distribution charging sessions using t-SNE, run the following command:

```
python .\tools\T_SNE_distribution_data.py acn_office attack -b -s -d
```

For more information:

```predict help
python .\tools\T_SNE_distribution_data.py -h
```
