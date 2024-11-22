# Lab Task: Data Exploration, Visualization, and Classification

This document outlines the steps for analyzing the dataset, visualizing key features, and implementing classification algorithms manually.

---

## **Coding Exercise 1: Data Exploration and Preprocessing**

### **1. Data Quality**

- **Common Data Quality Problems**
  - Missing values in key features.
  - Outliers affecting model performance.
  - Duplicate entries reducing data integrity.
  - Skewed data distributions.
  - Inconsistent or noisy data entries.

- **Exploratory Data Analysis (EDA)**
  - Identify feature distributions and relationships.
  - Detect patterns or trends in data.
  - Observe correlations between features.
  - Understand data skewness or imbalance.
  - Assess feature importance for target prediction.

- **Anomaly Detection**
  - Outliers in continuous features (e.g., extremely high/low values).
  - Missing or unexpected categories in categorical data.
  - Rare combinations of features.
  - Incorrect data types for certain columns.
  - Inconsistencies across similar records.

- **Summary Statistics**
  - Use `.describe()` to compute mean, median, min, max, and quartiles.
  - Summary highlights range and spread of values.
  - Identify skewness or data imbalance.
  - Outliers can be inferred from extreme values.
  - Distribution insights aid preprocessing decisions.

### **2. Data Visualization**

- **Visualizations**
  - **Histograms**: Show feature distributions.
  - **Scatter Plots**: Highlight relationships between two features.
  - **Contour Plots**: Visualize data density or clustering.
  - **Matrix Plots**: Represent pairwise relationships for all features.

- **Observations**
  - **Histograms**:  
    - **Pros**: Easy to identify distributions and outliers.  
    - **Cons**: Ineffective for high-dimensional data or relationships.  
  - **Scatter Plots**:  
    - **Pros**: Visualize relationships and clusters.  
    - **Cons**: Overlap in dense areas can obscure patterns.  
  - **Contour Plots**:  
    - **Pros**: Capture data density effectively.  
    - **Cons**: Requires well-tuned parameters for clarity.  
  - **Matrix Plots**:  
    - **Pros**: Compact visualization of feature relationships.  
    - **Cons**: Difficult to interpret with many features.  

---

## **Coding Exercise 2: Classification and Evaluation**

### **1. Nearest Neighbor Classifier**
```python
class NearestNeighborClassifierManual:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            distances = np.linalg.norm(self.X_train - x_test, axis=1)
            nearest_index = np.argmin(distances)
            predictions.append(self.y_train[nearest_index])
        return predictions


class GaussianNaiveBayesClassifierManual:
    def __init__(self):
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

    def fit(self, X_train, y_train):
        unique_classes = np.unique(y_train)
        for c in unique_classes:
            X_c = X_train[y_train == c]
            self.class_priors[c] = len(X_c) / len(y_train)
            self.class_means[c] = X_c.mean(axis=0)
            self.class_variances[c] = X_c.var(axis=0)

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            posteriors = []
            for c in self.class_priors:
                prior = np.log(self.class_priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.class_variances[c])) \
                             - 0.5 * np.sum(((x_test - self.class_means[c]) ** 2) / self.class_variances[c])
                posteriors.append(prior + likelihood)
            predictions.append(np.argmax(posteriors))
        return predictions



class SupportVectorMachineClassifierManual:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        n_features = X_train.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for x, y in zip(X_train, y_train):
                if y * (np.dot(x, self.weights) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.weights - np.dot(x, y))
                    self.bias -= self.learning_rate * y

    def predict(self, X_test):
        return np.sign(np.dot(X_test, self.weights) - self.bias)


class ConfusionMatrix:
    def __init__(self, y_true, y_pred, n_classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.n_classes = n_classes

    def compute_confusion_matrix(self):
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true, pred in zip(self.y_true, self.y_pred):
            matrix[true][pred] += 1
        return matrix

    def plot(self, matrix):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()


class EvaluationMetrics:
    def __init__(self, y_true, y_pred, confusion_matrix):
        self.y_true = y_true
        self.y_pred = y_pred
        self.confusion_matrix = confusion_matrix

    def compute_metrics(self):
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP
        FN = self.confusion_matrix.sum(axis=1) - TP
        TN = self.confusion_matrix.sum() - (FP + FN + TP)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "Precision": np.mean(precision),
            "Recall": np.mean(recall),
            "F1 Score": np.mean(f1_score),
        }


