# Svara Identification in Carnatic Music

## Introduction

This repository contains the implementation of software for the identification of svaras in the raga Bhairavi, a significant component of Carnatic music. This project is aimed at addressing the underrepresentation of non-Eurogenetic traditions in Music Information Retrieval (MIR). The software developed provides tools for feature extraction, model training, and svara identification, with a specific focus on the Carnatic musical piece "Sanjay Subrahmanyan - Kamakshi."

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
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

