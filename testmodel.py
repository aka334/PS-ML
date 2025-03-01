import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# List to hold the dataframes for new test files
dfs_test = []

# Loop through the new file numbers
for i in range(50, 60):
    file_path = f'./features_o{i}.txt'
    df_test = pd.read_csv(file_path)
    dfs_test.append(df_test)

# Concatenate all dataframes in the list
df_new_test = pd.concat(dfs_test, ignore_index=True)


# Adding statistical features across the hash counts
hash_columns = [f'Hash_Count_{i}' for i in range(1, 5)]
df_new_test['Mean_Hash_Counts'] = df_new_test[hash_columns].mean(axis=1)
df_new_test['Median_Hash_Counts'] = df_new_test[hash_columns].median(axis=1)
df_new_test['Std_Hash_Counts'] = df_new_test[hash_columns].std(axis=1)

df_new_test['Range_Class'] = df_new_test['Actual_Count'].apply(lambda x: '1-10000' if x <= 10000 else '10000-max')

#Step 1 Perform Classification

from joblib import load
classifier = load('random_forest_classifier.joblib')

features = df_new_test.drop(['Actual_Count', 'Range_Class','Flow_ID'], axis=1)
label_classifier = df_new_test['Range_Class']

predicted_classes = classifier.predict(features)

df_new_test['Predicted_Range_Class'] = predicted_classes

# Separate the data into two DataFrames based on the predicted range class
df_class_1_10000 = df_new_test[df_new_test['Predicted_Range_Class'] == '1-10000'].drop(['Predicted_Range_Class', 'Range_Class'], axis=1)
df_class_10000_max = df_new_test[df_new_test['Predicted_Range_Class'] == '10000-max'].drop(['Predicted_Range_Class', 'Range_Class'], axis=1)


#Perform Regression
@register_keras_serializable()
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    error = y_true - y_pred
    return tf.reduce_mean(tf.square(error) * tf.sqrt(1 + y_true))
 
# Now try loading the models again
model_1_10000 = load_model('model1-10000.keras', custom_objects={'custom_loss': custom_loss})
model_10000_max = load_model('model10000-max.keras', custom_objects={'custom_loss': custom_loss})

import joblib

scaler1 = joblib.load('scaler1-10000.joblib')
scaler2 = joblib.load('scaler10000-max.joblib')


# Preparing data for the first model (1-10000)
features_1_10000 = df_class_1_10000.drop(['Actual_Count', 'Flow_ID'], axis=1)
target_1_10000 = df_class_1_10000['Actual_Count']

# Preparing data for the second model (10000-max)
features_10000_max = df_class_10000_max.drop(['Actual_Count', 'Flow_ID'], axis=1)
target_10000_max = df_class_10000_max['Actual_Count']


features_1_10000_scaled = scaler1.transform(features_1_10000)
features_10000_max_scaled = scaler2.transform(features_10000_max)

# Making predictions using the loaded models
predictions_1_10000 = model_1_10000.predict(features_1_10000_scaled)
predictions_10000_max = model_10000_max.predict(features_10000_max_scaled)

# Adding predictions back to the original dataframes
df_class_1_10000['Prediction_ML'] = predictions_1_10000.flatten()  # Using flatten() to ensure the shape matches if necessary
df_class_10000_max['Prediction_ML'] = predictions_10000_max.flatten()

#Perform accuracy Measurment