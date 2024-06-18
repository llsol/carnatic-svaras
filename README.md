# Svara Identification in Carnatic Music

## Introduction

This repository contains the implementation of software for the identification of svaras in the raga Bhairavi, a significant component of Carnatic music. This project is aimed at addressing the underrepresentation of non-Eurogenetic traditions in Music Information Retrieval (MIR). The software developed provides tools for feature extraction, model training, and svara identification, with a specific focus on the Carnatic musical piece "Sanjay Subrahmanyan - Kamakshi."

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Main Code File](#main-code-file)
- [Supporting Scripts](#supporting-scripts)
- [Installation](#installation)
- [Datasets](#datasets)
- [Model Training](#model-training)
- [Transition Characterization and Modelling](#transition-characterization-and-modeling)
- [Further Development](#further-development)
- [Other Files in Repository](#other-files-in-repository)
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
f1-score: 0.30

### Second Dataset
Features: Basic features with context of preceding and following segments
f1-score: 0.37

### Third Dataset
Features: Basic features with labels for previous and next svaras
f1-score: 0.84

## Model Training
The model is a Gradient Boosting Classifier trained using the sklearn library. The datasets are split into training and testing sets, and missing values are handled using SimpleImputer. The training process includes saving the datasets to .csv files for reproducibility.

## Transition characterization and modeling

The following files consititute the current state of this research in the task of segmentation. 

transition_second_model_characterisation.py extracts features from audio segments and pairs them, labeling each pair based on whether there is a transition between them. The script focuses in transition detection.

transition_identification_model_second_method.py uses the features extracted and characterized by the transition_second_model_characterisation.py script. The model is trained and evaluated using a Gradient Boosting Classifier. Current best result is f1-score = 0.71.

## Further Development

### Svara Identification

- A comparative study with different tools for separating the main voice might have the potential to slightly improve the performance of the predictive model.
- A comparative and systematic study of different tools for extracting more complex features, such as the extraction of the main melody (for this, there are tools beyond what librosa offers), could also influence the result.
- A general collection of descriptors and their Spearman analysis, not only by feature but also by comparing their ability to differentiate each pair of svaras, could help select the most promising features.

### Svara Segmentation

- We have not yet tested methods based on the use of Dynamic Time Warping (DTW). It would be interesting to select 'model' samples of individual svaras and calculate their similarity, scanning through the piece. Additionally, with field knowledge, the different major versions of the svara could be selected and all used. This derives from the idea that while a svara experiences coarticulation effects, some versions seem to be more prevalent.
- In relation to what was mentioned just above, one way to segment could follow an order-based strategy. This would consist of seeing (if it happens) which svaras segment better with the previous method and successively segmenting the svaras that work best, leaving the most ambiguous ones for the end. This could yield better intermediate results.


## Other Files in Repository

All the files related to transition characterization and prediction have given negative results, so they will be removed soon.

- **transition_feature_visualizations.py**: Running this script shows the low potential of the features extracted and strategies tried so far to predict where transitions between svaras are. These had the aim to develop a segmentation pipeline.

- **svara_feature_spearman_rank_correlation_matrix.py**: This script shows the features used for the svara characterization.



## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.


## License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for more details.


