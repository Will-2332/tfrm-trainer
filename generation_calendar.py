import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load your data
data = pd.read_csv("dummy_calendar_data.csv")

# Check for NaN values
print(data.isna().sum())

# Remove rows with NaN values
data = data.dropna()

# Preprocessing

# Convert the string data in 'title' and 'location' to integer indices
title_encoder = {title: i for i, title in enumerate(data['title'].unique())}
location_encoder = {location: i for i, location in enumerate(data['location'].unique())}

data['title'] = data['title'].map(title_encoder)
data['location'] = data['location'].map(location_encoder)

# Here we'll just convert the times to timestamps and grade to integer
data['startTime'] = pd.to_datetime(data['startTime']).astype(np.int64) // 10**9
data['endTime'] = pd.to_datetime(data['endTime']).astype(np.int64) // 10**9
data['grade'] = data['grade'].astype(int)

# Create training and testing datasets
train, test = train_test_split(data, test_size=0.2)

# Same for train and test datasets
train['title'] = train['title'].map(title_encoder)
train['location'] = train['location'].map(location_encoder)

test['title'] = test['title'].map(title_encoder)
test['location'] = test['location'].map(location_encoder)

# Create a title model
title_model = keras.Sequential([
  layers.Embedding(len(data['title'].unique()) + 1, 32, input_length=1, name="title_embedding"),
  layers.Flatten(),
  layers.Dense(32, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(1)
])

# Create a location model
location_model = keras.Sequential([
  layers.Embedding(len(data['location'].unique()) + 1, 32, input_length=1, name="location_embedding"),
  layers.Flatten(),
  layers.Dense(32, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(1)
])

# Create a start time model
start_model = keras.Sequential([
  layers.Dense(32, activation='relu', input_shape=(1,)),
  layers.Dense(32, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(1)
])

# Create an end time model
end_model = keras.Sequential([
  layers.Dense(32, activation='relu', input_shape=(1,)),
  layers.Dense(32, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(1)
])

# Create a grade model
grade_model = keras.Sequential([
  layers.Dense(32, activation='relu', input_shape=(1,)),
  layers.Dense(32, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(1)
])

# Define the model
inputs = {
  'title': layers.Input(shape=(1,), dtype=tf.int32),
  'location': layers.Input(shape=(1,), dtype=tf.int32),
  'startTime': layers.Input(shape=(1,), dtype=tf.int64),
  'endTime': layers.Input(shape=(1,), dtype=tf.int64),
  'grade': layers.Input(shape=(1,), dtype=tf.int32),
}

title_embeddings = title_model(inputs['title'])
location_embeddings = location_model(inputs['location'])
start_embeddings = start_model(inputs['startTime'])
end_embeddings = end_model(inputs['endTime'])
grade_embeddings = grade_model(inputs['grade'])

x = layers.Concatenate()([title_embeddings, location_embeddings, start_embeddings, end_embeddings, grade_embeddings])
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  loss=tf.keras.losses.BinaryCrossentropy(),
  metrics=[keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name='auc')]
)

# Define target variable
target = 'suggestion'

# Separate input features from target variable for training data
train_features = train.drop(target, axis=1)
train_target = train[target]

# Separate input features from target variable for test data
test_features = test.drop(target, axis=1)
test_target = test[target]

# Convert input features to a format that TensorFlow can understand
train_features = {name: np.array(value) for name, value in train_features.items()}
test_features = {name: np.array(value) for name, value in test_features.items()}

# Train the model
model.fit(train_features, train_target, epochs=5)

# Evaluate the model
model.evaluate(test_features, test_target)
