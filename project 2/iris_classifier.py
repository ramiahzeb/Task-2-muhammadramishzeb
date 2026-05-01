# Step 0: Import the tools we need
from sklearn.datasets import load_iris                 # The iris dataset
from sklearn.model_selection import train_test_split   # To split data
from sklearn.preprocessing import StandardScaler       # For scaling
from sklearn.neighbors import KNeighborsClassifier     # The KNN model
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------------------
# 1. INPUT: Load the Iris dataset
# -------------------------------------------
iris = load_iris()
X = iris.data          # The measurements (150 samples × 4 features)
y = iris.target        # The species (0, 1, or 2)
target_names = iris.target_names  # ['setosa' 'versicolor' 'virginica']

print("Dataset shape:", X.shape)
print("First 5 samples:\n", X[:5])
print("First 5 labels:", y[:5])

# -------------------------------------------
# 2. PROCESS: Scaling
# -------------------------------------------
# StandardScaler subtracts the mean and divides by the standard deviation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # Transform all features

print("\nAfter scaling - first sample:", X_scaled[0])

# -------------------------------------------
# 3. PROCESS: Train-Test Split (80% train, 20% test)
# -------------------------------------------
# random_state makes the split reproducible (any number works)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y ensures each species is proportionally represented in both sets

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -------------------------------------------
# 4. PROCESS: Create and train the KNN model
# -------------------------------------------
# We'll start with K=5, as the PDF suggests
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)       # This is where the "learning" happens

# -------------------------------------------
# 5. OUTPUT: Make predictions on the test set
# -------------------------------------------
y_pred = model.predict(X_test)

# Compare predictions with the real answers (just first 10)
print("\nTrue labels    :", y_test[:10])
print("Predicted labels:", y_pred[:10])

# -------------------------------------------
# 6. VALIDATION: Confusion Matrix & F1 Score
# -------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Visualize it nicely
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix (K={k})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# F1 Score (weighted average across all three species)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\nWeighted F1 Score: {f1:.3f}")

# Full classification report (precision, recall, f1 per class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))