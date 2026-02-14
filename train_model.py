from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# ------------------------------------------------------------
# 1) Fetch dataset directly from UCI
# ------------------------------------------------------------
breast_cancer = fetch_ucirepo(id=15)

X = breast_cancer.data.features
y = breast_cancer.data.targets

# Convert labels (2 = benign, 4 = malignant)
y = y.replace({2: 0, 4: 1})

# ------------------------------------------------------------
# 2) Train/test split (25% test as required)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 3) Build Pipeline
# ------------------------------------------------------------
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# ------------------------------------------------------------
# 4) Train + evaluate
# ------------------------------------------------------------
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# ------------------------------------------------------------
# 5) Save model
# ------------------------------------------------------------
joblib.dump(pipe, "model.pkl")
print("Saved model as model.pkl")
print("Feature order:", list(X.columns))
