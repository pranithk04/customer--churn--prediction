import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Generate Synthetic Customer Data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'tenure_months': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20.0, 120.0, n_samples),
    'total_charges': np.random.uniform(100.0, 8000.0, n_samples),
    'support_calls': np.random.randint(0, 10, n_samples),
    'churn': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
})

# 2. Train-Test Split
X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Building & Evaluation
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))
