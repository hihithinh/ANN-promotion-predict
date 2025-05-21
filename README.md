# Employee Promotion Prediction - ANN and Random Forest

This project implements and compares two models for predicting employee promotion:
1.  An Artificial Neural Network (ANN) built from scratch (using NumPy).
2.  A Random Forest classifier (using scikit-learn).

The primary goal is to explore techniques for handling imbalanced datasets, such as SMOTE, and to evaluate model performance in predicting a minority class (employee promotion).

This project is inspired by and uses the dataset from the Kaggle notebook:
[Predicting Employee Promotion](https://www.kaggle.com/code/ifeanyichukwunwobodo/predicting-employee-promotion)

## Project Structure

```
.
├── data/                     # Placeholder for data files (not committed)
│   └── train.csv             # Training data (needs to be placed here manually)
├── ann_promotion_script.py   # Script for the ANN model
├── random_forest_promotion_script.py # Script for the Random Forest model
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── .gitignore                # Specifies intentionally untracked files
```

## Setup and Installation

### 1. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment (e.g., named .venv)
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

Once the virtual environment is activated, install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

This repository does not include the `train.csv` data file.
1.  Download `train.csv` from the original Kaggle dataset or your source.
2.  Create a directory named `data` in the root of the project.
3.  Place the `train.csv` file inside the `data/` directory.

## Running the Models

Make sure your virtual environment is activated and you are in the project's root directory.

### 1. Running the Artificial Neural Network (ANN)

This script uses SMOTE for handling class imbalance and trains an ANN from scratch.

```bash
python3 ann_promotion_script.py
```
The script will output preprocessing information, training progress (cost per iteration), evaluation metrics (accuracy, classification report, confusion matrix) for both training and validation sets, and a learning curve plot.

### 2. Running the Random Forest Model

This script uses `class_weight='balanced'` to handle class imbalance.

```bash
python3 random_forest_promotion_script.py
```
The script will output preprocessing information, and evaluation metrics (accuracy, classification report, confusion matrix) for both training and validation sets.

## Notes

*   The ANN script (`ann_promotion_script.py`) includes an implementation of SMOTE and is currently configured to train for 7500 iterations.
*   The Random Forest script (`random_forest_promotion_script.py`) uses the `class_weight='balanced'` parameter.
*   The Jupyter Notebook (`predicting-employee-promotion.ipynb`) from which this project was inspired is not included in this repository but can be accessed at the Kaggle link above.
