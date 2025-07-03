# Sensor-Based Risk Assessment

**Note:** The main Jupyter notebook (`sensor-based-risk-assessment-for-manufacturing.ipynb`) and the `requirements.txt` file are both available in the files section of this repository.

## Overview

This project provides a comprehensive pipeline for risk assessment in manufacturing using sensor data and deep learning. The main entry point is the Jupyter Notebook (`sensor-based-risk-assessment-for-manufacturing.ipynb`), which processes time-series sensor data, engineers features, and applies a variety of machine learning and deep learning models—including CNNs, RNNs, and ensemble methods—to predict risk levels for manufacturing units.

## Features

- **Data Loading:** Reads and preprocesses sensor data from CSV files (train, test, and sample submission).
- **Feature Engineering:** Extracts and transforms sensor readings for model input, including flattening and reshaping for neural networks.
- **Visualization:** Plots time-series sensor data colored by risk category for exploratory analysis.
- **Modeling:** Implements and compares several models:
  - Conv1D + LSTM
  - Conv1D + GRU
  - Dense Neural Network
  - SimpleRNN
  - Conv2D CNN (on reshaped sensor data as images)
  - AutoGluon Tabular Predictor (AutoML)
  - Classical ML models (Random Forest, Decision Tree, Logistic Regression, SVM, LightGBM, XGBoost, AdaBoost)
- **Image Conversion:** (Commented) Converts sensor data segments into grayscale images for CNN training.
- **Ensembling:** Combines predictions from multiple models using hard voting for robust risk classification.
- **Submission:** Generates a submission file in the required format for competitions.

## File Structure

- `sensor-based-risk-assessment-for-manufacturing.ipynb` — Main Jupyter Notebook containing all data processing, modeling, and analysis steps (recommended entry point). **Available in the files section.**
- `readme.md` — Project documentation (this file).
- `requirements.txt` — List of required Python packages. **Available in the files section.**

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Kaggle API (for dataset download, or place data files manually)
- Common data science libraries:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `opencv-python`
  - Deep learning: `tensorflow`, `keras`
  - AutoML: `autogluon`
  - Gradient boosting: `lightgbm`, `xgboost`

All required packages are listed in `requirements.txt` in the files section of this repository.

Install requirements with:

```bash
pip install -r requirements.txt
```

### Data

The project uses the following dataset from Kaggle:

- [Sensor Based Risk Dataset](https://www.kaggle.com/datasets/ramakrishna50/sensorbasedriskdataset)

The notebook expects the following CSV files (paths may need adjustment):

- `train.csv` — Training data with sensor readings and risk labels
- `test.csv` — Test data for prediction
- `Sample_Submission Kaggle.csv` — Sample submission format

### Usage

1. Download the dataset from Kaggle and place the required CSV files in the appropriate directory (or update the file paths in the notebook).
2. Open the notebook:
   ```bash
   jupyter notebook sensor-based-risk-assessment-for-manufacturing.ipynb
   ```
3. Run the cells sequentially to process data, train models, and view results.

The notebook will process the data, train multiple models, ensemble their predictions, and generate a `submission.csv` file.

## Models Implemented

- **Conv1D + LSTM:** 1D convolutional layer followed by LSTM for sequential sensor data.
- **Conv1D + GRU:** 1D convolutional layer followed by GRU.
- **Dense Neural Network:** Fully connected layers for tabular sensor features.
- **SimpleRNN:** Simple recurrent neural network for sequence modeling.
- **Conv2D CNN:** Treats reshaped sensor data as images for 2D convolution.
- **AutoGluon Tabular Predictor:** Automated machine learning for tabular data.
- **Classical ML Models:** Decision Tree, Random Forest, Logistic Regression, SVM, LightGBM, XGBoost, AdaBoost.
- **Ensemble Voting:** Combines predictions from all major models for final risk classification.

## Risk Categories

The models predict risk as one of four categories:

- 0: No Risk
- 1: Low Risk
- 2: Medium Risk
- 3: Catastrophic

## Output

- `submission.csv` — Contains the predicted risk state for each test sample, formatted for competition submission.

## Notes

- Some sections (image conversion, augmentation, etc.) are provided as commented code for further experimentation.
- The notebook is designed for flexibility and can be extended with additional models or feature engineering steps.
- All requirements for running the notebook are listed in `requirements.txt` in the files section of this repository.
