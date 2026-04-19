import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# --- load MATLAB-extracted features ---
df = pd.read_csv("Model_Ready_Features_Fixed.csv")
print("Dataset loaded:")
print(df['label'].value_counts())

FEATURE_COLS = [
    'mean_aX','mean_aY','mean_aZ','mean_gX','mean_gY','mean_gZ',
    'std_aX', 'std_aY', 'std_aZ', 'std_gX', 'std_gY', 'std_gZ',
    'rms_aX', 'rms_aY', 'rms_aZ', 'rms_gX', 'rms_gY', 'rms_gZ',
    'sma_acc','sma_gyro'
]

X = df[FEATURE_COLS].values
y = df['label'].values

# --- train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

# --- train ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- evaluate ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv  = cross_val_score(model, X, y, cv=5).mean()

print(f"\nTest accuracy : {acc*100:.1f}%")
print(f"CV accuracy   : {cv*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred,
      labels=['zone2','aerobic','tempo','intense']))

# --- save ---
joblib.dump(model, "zone_classifier.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")
print("\nSaved: zone_classifier.pkl")
print("Saved: feature_cols.pkl")