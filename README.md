# Coding-Generation-Project 🦾

## Predictive Maintenance Using Machine Learning
A supervised classification project using the AI4I 2020 Predictive Maintenance dataset. 

## Overview
This project investigates predictive maintenance as a machine learning task. The aim is to predict whether an industrial machine will fail using sensor readings and engineered features derived from the AI4I 2020 Predictive Maintenance dataset.

The project also compares AI-generated Python code produced by **Claude Sonnet 4.6 (Extended)** and **ChatGPT 5.4 (Thinking)**. Both models were used to generate initial machine learning pipelines, which were then reviewed, validated, and refined.

## Objectives
- Predict machine failure from industrial sensor data
- Compare multiple supervised learning algorithms
- Evaluate the strengths and weaknesses of AI-generated code
- Improve model reliability through preprocessing and feature engineering

## Dataset
The dataset used is the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

### Input features
- Air temperature [K]
- Process temperature [K]
- Rotational speed [RPM]
- Torque [Nm]
- Tool wear [min]

### Target
- Machine failure (binary classification)

## Failure Modes
The dataset includes five failure modes:
- Tool Wear Failure (TWF)
- Random Failure (RNF)
- Heat Dissipation Failure (HDF)
- Power Failure (PWF)
- Overstrain Failure (OSF)

## Models Compared
This project compares four supervised classifiers:
- Gaussian Naïve Bayes
- Logistic Regression
- SVM with RBF kernel
- Random Forest

## Feature Engineering
Three additional features were engineered:
- Mechanical power = torque × 2π × rotational speed / 60
- Wear–torque interaction = tool wear × torque
- Temperature differential = process temperature − air temperature

The product type column was also one-hot encoded.

## Preprocessing
The final pipeline includes:
- 80/20 train-test split
- SMOTE applied only to the training set
- StandardScaler fitted on training data only
- Evaluation using precision, recall, F1-score, and ROC-AUC

## Results
### Best overall model
**Random Forest**
- F1-score: 0.770
- Precision: 0.713
- ROC-AUC: 0.984

### Highest recall
**SVM with RBF kernel**
- Recall: 0.853

### Key conclusion
Random Forest achieved the strongest overall balance of performance metrics, while SVM with an RBF kernel was strongest for recall.

## AI-Assisted Coding Reflection
Claude and ChatGPT both generated functional code, but the outputs required careful validation. This project highlights that AI-generated code can accelerate development, but plausible-looking outputs must still be reviewed line by line using existing programming knowledge.

## Repository Structure
```text

├── ChatGPT/
│   ├── Initial
│   └── Final
│ 
├── Claude/
│   ├── Initial
│   └── Final
│   
└── README.md
