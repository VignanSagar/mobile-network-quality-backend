import pandas as pd

# Load the CSV dataset
df = pd.read_csv("../dataset/network_quality_dataset.csv",encoding="cp1252")

# Display basic information
print("===== DATASET INFORMATION =====")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== UNIQUE VALUES IN CATEGORICAL COLUMNS =====")

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print("\n", column)
    print(df[column].unique())

# Create Network Quality Score

df["Quality_Score"] = (
    0.30 * df["Signal Strength (dBm)"].rank(pct=True) +
    0.25 * df["Download Speed (Mbps)"].rank(pct=True) +
    0.10 * df["Upload Speed (Mbps)"].rank(pct=True) +
    0.15 * (1 - df["Latency (ms)"].rank(pct=True)) +
    0.10 * (1 - df["Jitter (ms)"].rank(pct=True)) +
    0.10 * (1 - df["Ping to Google (ms)"].rank(pct=True))
) * 100

# Convert score into quality classes

def classify_quality(score):
    if score >= 70:
        return "GOOD"
    elif score >= 40:
        return "MODERATE"
    else:
        return "POOR"

df["Network_Quality"] = df["Quality_Score"].apply(classify_quality)

print("\n===== NETWORK QUALITY =====")
print(df["Network_Quality"].value_counts())

print("\n===== SAMPLE PREDICTIONS =====")
print(df[["Quality_Score", "Network_Quality"]].head(10))

print("\n===== QUALITY SCORE DISTRIBUTION =====")
print(df["Quality_Score"].describe())

print("\n===== QUALITY SCORE PERCENTILES =====")
print(df["Quality_Score"].quantile([0.33, 0.50, 0.67]))

# Create balanced Network Quality classes

poor_threshold = df["Quality_Score"].quantile(0.33)
good_threshold = df["Quality_Score"].quantile(0.67)

def classify_quality(score):
    if score <= poor_threshold:
        return "POOR"
    elif score <= good_threshold:
        return "MODERATE"
    else:
        return "GOOD"

df["Network_Quality"] = df["Quality_Score"].apply(classify_quality)

print("\n===== FINAL NETWORK QUALITY DISTRIBUTION =====")
print(df["Network_Quality"].value_counts())

print("\n===== PERCENTAGE DISTRIBUTION =====")
print(df["Network_Quality"].value_counts(normalize=True).mul(100).round(2))

# ==========================================
# STEP 4: PREPARE DATA FOR MACHINE LEARNING
# ==========================================

features = [
    "Signal Strength (dBm)",
    "Download Speed (Mbps)",
    "Upload Speed (Mbps)",
    "Latency (ms)",
    "Jitter (ms)",
    "Network Type",
    "Carrier",
    "Band",
    "Network Congestion Level",
    "Ping to Google (ms)",
    "Handover Count",
    "Dropped Connection"
]

target = "Network_Quality"

X = df[features]
y = df[target]

print("\n===== ML FEATURES =====")
print(X.columns.tolist())

print("\n===== TARGET =====")
print(y.name)

print("\n===== FEATURE DATA SHAPE =====")
print(X.shape)

print("\n===== TARGET DATA SHAPE =====")
print(y.shape)

# ==========================================
# STEP 4.2: ENCODE CATEGORICAL FEATURES
# ==========================================

X_encoded = pd.get_dummies(X, drop_first=False)

print("\n===== ENCODED FEATURES =====")
print(X_encoded.columns.tolist())

print("\n===== ENCODED DATA SHAPE =====")
print(X_encoded.shape)

print("\n===== CHECK FOR MISSING VALUES =====")
print(X_encoded.isnull().sum().sum())

print("\n===== COLUMNS WITH MISSING VALUES =====")
print(X_encoded.isnull().sum()[X_encoded.isnull().sum() > 0])

# ==========================================
# STEP 4.3: HANDLE MISSING VALUES
# ==========================================

numerical_features = [
    "Signal Strength (dBm)",
    "Download Speed (Mbps)",
    "Upload Speed (Mbps)",
    "Latency (ms)",
    "Jitter (ms)",
    "Handover Count"
]

for column in numerical_features:
    X_encoded[column] = X_encoded[column].fillna(
        X_encoded[column].median()
    )

print("\n===== MISSING VALUES AFTER IMPUTATION =====")
print(X_encoded.isnull().sum().sum())

# ==========================================
# STEP 4.4: TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n===== TRAINING AND TESTING DATA =====")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\n===== TRAINING CLASS DISTRIBUTION =====")
print(y_train.value_counts())

print("\n===== TESTING CLASS DISTRIBUTION =====")
print(y_test.value_counts())

# ==========================================
# STEP 5: RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# STEP 6: DECISION TREE MODEL
# ==========================================

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy:", round(dt_accuracy * 100, 2), "%")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, dt_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, dt_pred))

# ==========================================
# STEP 6.2: LOGISTIC REGRESSION MODEL
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Logistic Regression model
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train
lr_model.fit(X_train_scaled, y_train)

# Predict
lr_pred = lr_model.predict(X_test_scaled)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Accuracy:", round(lr_accuracy * 100, 2), "%")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, lr_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, lr_pred))

# ==========================================
# STEP 7: SAVE FINAL MODEL
# ==========================================

import joblib

# Save Logistic Regression model
joblib.dump(lr_model, "logistic_regression_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Save feature column names
joblib.dump(X_encoded.columns.tolist(), "feature_columns.pkl")

print("\n===== MODEL SAVED =====")
print("logistic_regression_model.pkl")
print("scaler.pkl")
print("feature_columns.pkl")