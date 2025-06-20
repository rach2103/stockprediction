import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# === Load Dataset ===
df = pd.read_csv('Tesla.csv')  # Make sure Tesla.csv is in the same folder

# === Basic Exploration ===
print(df.head())
print(df.describe())
print(df.info())

# === Line plot of Close Price ===
plt.figure(figsize=(15,5))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.title('Tesla Stock - Close Price')
plt.xlabel('Time')
plt.ylabel('Price in USD')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Drop 'Adj Close' if it's same as 'Close' ===
if 'Adj Close' in df.columns and df['Close'].equals(df['Adj Close']):
    df = df.drop(['Adj Close'], axis=1)

# === Check for Nulls ===
print("Missing values:\n", df.isnull().sum())

# === Distribution of Features ===
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.figure(figsize=(18, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# === Boxplots for Outliers ===
plt.figure(figsize=(18, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.boxplot(y=df[col], color='orange')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# === Define Target ===
df['Price_Up'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df = df.dropna()

# === Feature and Label Split ===
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Price_Up']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Models ===
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# === Training & Evaluation ===
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    acc = metrics.accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {acc:.2f}")
    print(metrics.classification_report(y_test, predictions))
    print("-" * 40)

# === Confusion Matrix for Best Model (XGBoost) ===
best_model = models["XGBoost"]
y_pred = best_model.predict(X_test_scaled)
cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()