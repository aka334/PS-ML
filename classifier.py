import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from joblib import dump, load

from sklearn.metrics import roc_auc_score, mean_squared_error
# List to hold the dataframes
dfs = []

# Loop through the file numbers
for i in range(1, 50):  # 1 to 10, inclusive
    file_path = f'./features_o{i}.txt'
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all dataframes in the list
df = pd.concat(dfs, ignore_index=True)

# Adding statistical features across the hash counts
hash_columns = [f'Hash_Count_{i}' for i in range(1, 5)]
df['Mean_Hash_Counts'] = df[hash_columns].mean(axis=1)
df['Median_Hash_Counts'] = df[hash_columns].median(axis=1)
df['Std_Hash_Counts'] = df[hash_columns].std(axis=1)

df['Range_Class'] = df['Actual_Count'].apply(lambda x: '1-10000' if x <= 10000 else '10000-max')

X_class = df.drop(['Actual_Count', 'Range_Class','Flow_ID'], axis=1)
y_class = df['Range_Class']


X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.4, random_state=42) #adjust test size accordingly

# Training the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100,           # Number of trees in the forest
    max_depth=20,               # Limiting depth of each tree
    min_samples_split=20,       # Minimum number of samples required to split an internal node
    min_samples_leaf=6,         # Minimum number of samples required to be at a leaf node
    max_features='sqrt',        # Maximum number of features considered for splitting a node
    random_state=42,
    n_jobs=20
)
rf_classifier.fit(X_train_class, y_train_class)

y_pred_rf_class = rf_classifier.predict(X_test_class)

print("Random Forest Classifier AUC:", roc_auc_score(y_test_class, rf_classifier.predict_proba(X_test_class)[:, 1]))
cm_rf = confusion_matrix(y_test_class, y_pred_rf_class)
print("Confusion Matrix:\n", cm_rf)
print("\nClassification Report:\n", classification_report(y_test_class, y_pred_rf_class))


# Save the model to disk
filename = 'random_forest_classifier.joblib'
dump(rf_classifier, filename)