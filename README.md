# Svara Identification in Carnatic Music

## Introduction

This repository contains the implementation of software for the identification of svaras in the raga Bhairavi, a significant component of Carnatic music. This project is aimed at addressing the underrepresentation of non-Eurogenetic traditions in Music Information Retrieval (MIR). The software developed provides tools for feature extraction, model training, and svara identification, with a specific focus on the Carnatic musical piece "Sanjay Subrahmanyan - Kamakshi."

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Main Code File](#main-code-file)
- [Supporting Scripts](#supporting-scripts)
- [Installation](#installation)
- [Main Code File](#main)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

## Features

- Identification of svaras in raga Bhairavi using MIR techniques
- Feature extraction from audio files
  - Pitch curve features
  - Time domain features
  - Frequency domain features
- Training a Gradient Boosting Classifier for svara identification
- Three datasets with different feature sets for model training
- User-friendly interface for feature extraction and model testing

## Main Code File

The main code file for this project is `interface.py`. This script creates a user-friendly interface for audio processing, utilizing customtkinter for the graphical interface. It allows users to select audio and text files, extract features with context or labels, and download the resulting CSV files.

### Key Functionality

- **Audio and Annotations Input**: The main purpose of `interface.py` is to receive a separated voice audio file and an annotations file. The interface processes these inputs to extract relevant features for svara identification.
- **File Selection Tab**: Allows the user to select audio and text files for analysis.
- **Feature Extraction Tab**: Provides options to extract features with context or labels and download the resulting CSV files.

The `interface.py` script integrates functionalities from the following scripts:
- `svara_characterisation_2_context.py`: For extracting features with context.
- `svara_characterisation_3_labels.py`: For extracting features with labels.

## Supporting Scripts

### `svara_characterisation_2_context.py`

This script is used to extract features with context from the audio file. It processes the audio data and outputs a CSV file containing the extracted features.

### `svara_characterisation_3_labels.py`

This script extracts features with labels from the audio file. Similar to the context script, it processes the audio data and outputs a CSV file containing the labeled features.

## Additional Preprocessing Step

### `svara-segmentation.ipynb`

The file `svara-segmentation.ipynb` has been used in Google Colab to get the separated voice file. This is a necessary preprocessing step when working with another piece. The notebook processes the audio to separate the voice from other components, preparing it for further analysis by the main interface. 

Further improvements to the project could include integrating this preprocessing step into the main pipeline to streamline the workflow.


## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/llsol/carnatic-svaras.git
    cd carnatic-svaras
    ```


2. You can install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```



## Datasets

### First Dataset
Features: Basic features without context or labels
Accuracy: 30%

### Second Dataset
Features: Basic features with context of preceding and following segments
Accuracy: 37%

### Third Dataset
Features: Basic features with labels for previous and next svaras
Accuracy: 84%

## Model Training
The model is a Gradient Boosting Classifier trained using the sklearn library. The datasets are split into training and testing sets, and missing values are handled using SimpleImputer. The training process includes saving the datasets to .csv files for reproducibility.

