import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# For nicer plots
sns.set(style="whitegrid")
%matplotlib inline
# Import data from CSV file

df = pd.read_csv("eda_classification.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
df.head(3)

# Check balance
print(df['y'].value_counts(normalize=True).round(3) * 100)
sns.countplot(x='y', data=df)
plt.title("Target Distribution")
plt.show()

# Standardize month names
month_map = {
    'sept.': 'Sep', 'sept': 'Sep', 'Dev': 'Dec', 'thurday': 'Thursday',
    'wed': 'Wednesday', 'thur': 'Thursday', 'tuesday': 'Tuesday',
    'wednesday': 'Wednesday', 'friday': 'Friday'
}
df['x1'] = df['x1'].replace(month_map)
df['x14'] = df['x14'].replace(month_map)  # typo in original, but just in case

# Fix brand names
brand_map = {'volkswagon': 'volkswagen', 'chrystler': 'chrysler'}
df['x13'] = df['x13'].replace(brand_map)

# Fix size
df['x17'] = df['x17'].str.lower()  # just in case

# Parse x7: currency strings like "($1,306.52)" or "$1,213.37 "
def clean_currency(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('$', '').replace('(', '-').replace(')', '').replace(',', '')
    try:
        return float(val)
    except:
        return np.nan

df['x7_clean'] = df['x7'].apply(clean_currency)

# Parse x11: percentages like "0.01%", "-0.01%", "0.00%"
def clean_percent(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('%', '')
    try:
        return float(val) / 100
    except:
        return np.nan

df['x11_clean'] = df['x11'].apply(clean_percent)

# Drop originals if you want (optional)
# df = df.drop(columns=['x7', 'x11'])

#Check for missing values after parsing
print("Missing values:\n", df.isnull().sum().sort_values(ascending=False).head(10))

# Drop rows with any missing (very few)
df = df.dropna()
print("\nShape after dropna:", df.shape)

#EDA numerical features correlatoin w/target
num_cols = ['x0','x2','x3','x4','x5','x6','x7_clean','x8','x9','x10','x11_clean','x12','x15','x16']
corr = df[num_cols + ['y']].corr()['y'].sort_values(ascending=False)
print("Correlation with y:\n", corr.round(4))

# Heatmap (optional – can be slow with many cols)
# plt.figure(figsize=(10,8))
# sns.heatmap(df[num_cols + ['y']].corr(), annot=False, cmap='coolwarm')
# plt.title("Feature Correlations")
# plt.show()

#Categorical Features - mean target by category
cat_cols = ['x1', 'x13', 'x14', 'x17']

for col in cat_cols:
    print(f"\n{'y'} rate by {col}:")
    print(df.groupby(col)['y'].mean().sort_values(ascending=False).round(3))
	
	# Define feature columns (using cleaned versions)
features = ['x0','x2','x3','x4','x5','x6','x7_clean','x8','x9','x10',
            'x11_clean','x12','x15','x16','x1','x13','x14','x17']

X = df[features]
y = df['y']

print("X shape:", X.shape)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

#Build preprocessing + Model Pipeline
numeric_features = ['x0','x2','x3','x4','x5','x6','x7_clean','x8','x9','x10',
                    'x11_clean','x12','x15','x16']
categorical_features = ['x1','x13','x14','x17']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # no scaling needed for RF
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

#evaluate
y_pred = rf_pipeline.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Logistic REgression for comparison
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

logreg_pipeline.fit(X_train, y_train)
y_pred_lr = logreg_pipeline.predict(X_test)

print("Logistic Regression Accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Feature importance from random forest
# Get feature names after one-hot encoding
feature_names = numeric_features + list(
    rf_pipeline.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)

importances = rf_pipeline.named_steps['classifier'].feature_importances_
imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("Top 15 features by importance:")
print(imp_df.head(15).round(4))

# Plot
plt.figure(figsize=(10,6))
imp_df.head(20).plot(kind='barh')
plt.title("Top 20 Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

# define features and split data.
# Numeric features including cleaned ones
numeric_features = ['x0', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7_clean', 'x8', 'x9', 'x10', 'x11_clean', 'x12', 'x15', 'x16']

# Categorical features
categorical_features = ['x1', 'x13', 'x14', 'x17']

# Target
X = df[numeric_features + categorical_features]
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# train baseline models
# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Random Forest Pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

rf_pipeline.fit(X_train, y_train)

# Logistic Regression Pipeline
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

logreg_pipeline.fit(X_train, y_train)

# evaluate with classification metrics
# Predictions
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_logreg = logreg_pipeline.predict(X_test)

# Confusion Matrix for Random Forest
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Classification Report for Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Accuracy for Random Forest
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Confusion Matrix for Logistic Regression
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))

# Classification Report for Logistic Regression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Accuracy for Logistic Regression
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")

# Perform tuning with hyperparameters and cross-validation
from sklearn.model_selection import GridSearchCV

# Tune Random Forest
rf_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"Best RF Params: {rf_grid.best_params_}")
print(f"Best RF CV F1-Score: {rf_grid.best_score_:.4f}")

# Evaluate tuned RF on test set
y_pred_rf_tuned = rf_grid.predict(X_test)
print("\nTuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf_tuned))
print(f"Tuned Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf_tuned):.4f}")

# Tune Logistic Regression
logreg_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__penalty': ['l2']  # l1 for liblinear only, but keeping simple
}

logreg_grid = GridSearchCV(logreg_pipeline, logreg_param_grid, cv=5, scoring='f1', n_jobs=-1)
logreg_grid.fit(X_train, y_train)

print(f"\nBest LR Params: {logreg_grid.best_params_}")
print(f"Best LR CV F1-Score: {logreg_grid.best_score_:.4f}")

# Evaluate tuned LR on test set
y_pred_logreg_tuned = logreg_grid.predict(X_test)
print("\nTuned Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg_tuned))
print(f"Tuned Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logreg_tuned):.4f}")
