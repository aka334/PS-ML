import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import joblib
# List to hold the dataframes
dfs = []

# Loop through the file numbers
for i in range(1, 50): 
    file_path = f'./features_o{i}.txt'
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all dataframes in the list
df = pd.concat(dfs, ignore_index=True)

hash_columns = [f'Hash_Count_{i}' for i in range(1, 5)]
df['Mean_Hash_Counts'] = df[hash_columns].mean(axis=1)
df['Median_Hash_Counts'] = df[hash_columns].median(axis=1)
df['Std_Hash_Counts'] = df[hash_columns].std(axis=1)
df['Range_Class'] = df['Actual_Count'].apply(lambda x: '1-10000' if x <= 10000 else '10000-max')

# Splitting the dataset into the two categories
df_low = df[df['Range_Class'] == '1-10000']
df_high = df[df['Range_Class'] == '10000-max']

df_high['Actual_Count'].describe()


features = df_high.drop(['Actual_Count', 'Flow_ID','Range_Class'], axis=1)
target = df_high['Actual_Count']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32) 
    error = y_true - y_pred
    return tf.reduce_mean(tf.square(error) * tf.sqrt(1 + tf.cast(y_true, tf.float32)))


early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=3,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the lowest validation loss
)

model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])

model.save('model10000-max.keras')  # Saves the model to a HDF5 file
joblib.dump(scaler, 'scaler10000-max.joblib')