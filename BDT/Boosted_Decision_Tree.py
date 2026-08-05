import pandas as pd

from sklearn.model_selection import cross_val_score, GroupKFold, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv(r"C:\Users\joshy\OneDrive\Documents\Kravitz Lab\freezeIdentifier\BDT\RQs.csv")

# Creates feature matrix

# X = df[
#     [
#         "pixel_std_0.0",
#         "entropy_0.0",
#         "laplace_var_0.0",
#         "brightness_0.0",
#         "ssim_0.0",
#         "pixel_std_2.0",
#         "entropy_2.0",
#         "laplace_var_2.0",
#         "brightness_2.0",
#         "ssim_2.0",
#         "pixel_std_4.0",
#         "entropy_4.0",
#         "laplace_var_4.0",
#         "brightness_4.0",
#         "ssim_4.0",
#         "pixel_std_6.0",
#         "entropy_6.0",
#         "laplace_var_6.0",
#         "brightness_6.0",
#         "ssim_6.0",
#         "pixel_std_8.0",
#         "entropy_8.0",
#         "laplace_var_8.0",
#         "brightness_8.0",
#         "ssim_8.0",
#         "pixel_std_10.0",
#         "entropy_10.0",
#         "laplace_var_10.0",
#         "brightness_10.0",
#         "ssim_10.0"
#     ]
# ]

X = df[
    [
        'pixel_std',
        'entropy',
        'laplace_var',
        'brightness',
        'ssim'
    ]
]

y = df["quality"]

# # Splits data into training and testing sets

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y,
# )

# Creates a BDT

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

# Sets up cross-validation

cv = GroupKFold(n_splits=5)
scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    groups=df["freezeID"],
    scoring="accuracy",
)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Train on all available data

model.fit(X, y)

# Shows relative importance of RQs to classification

importance = model.feature_importances_

plt.bar(X.columns, importance)
plt.xticks(rotation=45)
plt.ylabel("Importance")
plt.show()

# Saves the model
joblib.dump(model, "freeze_classifier.pkl")

print(
    df.groupby("freezeID")["quality"]
      .value_counts()
      .unstack(fill_value=0)
)