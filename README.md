
---

# Parkinson's Disease Detection Model

This repository contains a Jupyter Notebook for building and evaluating a machine learning model to detect Parkinson's disease based on patient voice measurements. The model uses multiple classifiers and combines their predictions to provide a final diagnosis.

## Summary

The notebook provides a comprehensive analysis of Parkinson's disease detection using voice measurements. The workflow includes:

1. Data loading and exploration.
2. Preprocessing steps to clean and prepare the data.
3. Training and evaluating various machine learning models.
4. Selecting the best model based on performance metrics.
5. Saving the trained model for future use.

## Data Source

The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/datasets/nidaguler/parkinson-disease-detection).

### Dataset Description

The dataset includes the following features:

- **MDVP:Fo(Hz)** - Average vocal fundamental frequency
- **MDVP:Fhi(Hz)** - Maximum vocal fundamental frequency
- **MDVP:Flo(Hz)** - Minimum vocal fundamental frequency
- **MDVP:Jitter(%), MDVP:Jitter(Abs), RAP, PPQ, DDP** - Several measures of variation in fundamental frequency
- **MDVP:Shimmer, MDVP:Shimmer(dB), APQ3, APQ5, APQ, DDA** - Several measures of variation in amplitude
- **NHR, HNR** - Measures of ratio of noise to tonal components in the voice
- **status** - Health status of the subject (one) - Parkinson's, (zero) - healthy
- **RPDE, DFA** - Two nonlinear dynamical complexity measures
- **spread1, spread2, PPE** - Three nonlinear measures of fundamental frequency variation

## Installation

To run this notebook, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install the necessary packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, you can use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone [<repository-url>](https://github.com/Sahiru2007/Parkinson-s-Disease-Detection-Model.git)
cd Parkinson-s-Disease-Detection-Model
```

2. Open the Jupyter Notebook:

```bash
jupyter notebook parkinson_disease_detection.ipynb
```

3. Execute all cells to reproduce the analysis and model training.

## Data Preprocessing

### Handling Missing Values

The dataset is cleaned by handling missing values, which are replaced by the median value of the respective columns.

### Feature Scaling

Standardization of features is performed to bring all attributes to a similar scale, essential for many machine learning algorithms.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Model Building

### Models Evaluated

The notebook evaluates multiple machine learning models, including:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

### Training Example

Example of training a Random Forest classifier:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
```

## Model Evaluation

### Evaluation Metrics

The models are evaluated using various metrics:

- **Accuracy**: The proportion of correctly classified instances.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A table to describe the performance of the classification model.

### Example: Evaluating Random Forest Classifier

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{cm}')
```

### Results

- **Logistic Regression**: Accuracy ~ 85%
- **Random Forest**: Accuracy ~ 88%
- **SVM**: Accuracy ~ 84%
- **KNN**: Accuracy ~ 82%

## Unique Aspects

- **Feature Importance**: Visualizing the importance of each feature in predicting Parkinson's disease.
- **Correlation Matrix**: Visualizing correlations between features.
- **ROC Curve**: Evaluating the trade-off between sensitivity and specificity.

### Feature Importance

```python
import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
```

### Correlation Matrix

```python
import seaborn as sns

corr_matrix = X.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc

y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

## Saving the Model

The trained model is saved using `pickle` for future use:

```python
import pickle

filename = 'parkinson_disease_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rf_model, file)

print(f"Model saved to {filename}")
```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---
