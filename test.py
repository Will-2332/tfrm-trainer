import numpy as np
import tensorflow as tf
import pandas as pd

print(tf.__version__)

print("Loading the saved model...")
loaded_model = tf.keras.models.load_model('TFRS_LTSM_model')
print("Model loaded successfully.")

print("Loading CSV file...")
df = pd.read_csv('enhanced_realistic_calendar_data.csv')
print(f"Loaded {len(df)} records from the CSV file.\n")
print("First few rows of the dataset:\n", df.head())

df['suggestion'] = 0
df['grade'] = 0

# Assuming df is your DataFrame for inference
print("Preprocessing datetime columns...")
df["startTime"] = pd.to_datetime(df["startTime"])
df["endTime"] = pd.to_datetime(df["endTime"])

print("Converting datetime to timestamps...")
df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())

# Convert the DataFrame to a TensorFlow Dataset
print("Converting DataFrame to TensorFlow Dataset...")
inference_dict = {name: np.array(value) for name, value in df.items()}
inference_ds = tf.data.Dataset.from_tensor_slices((inference_dict, inference_dict.pop("suggestion")))
inference_ds = inference_ds.batch(64, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

# Initialize an empty list to store the predictions
all_predictions = []

# Run inference
print("Running inference...")
for features, _ in inference_ds:
    try:
        predictions = loaded_model.predict(features)
        all_predictions.extend(predictions)
    except Exception as e:
        print(f"Exception during prediction: {e}")

# Check the length of all_predictions
print(f"Length of all_predictions: {len(all_predictions)}")

# Print model summary
print(loaded_model.summary())

# Convert predictions to a numpy array for easier manipulation
all_predictions = np.array(all_predictions)

# Flatten the all_predictions array
all_predictions = all_predictions.flatten()

n_recommendations = min(5, len(all_predictions))
if n_recommendations > 0:
    top_indices = all_predictions.argsort()[-n_recommendations:][::-1]
else:
    print("No recommendations available.")
    exit(0)


# Fetch the details of the top recommended activities from the original DataFrame
top_recommendations = df.iloc[top_indices]

print("Top Recommended Activities:")
for i, idx in enumerate(top_indices):
    row = df.iloc[idx]
    print(f"Sample {i + 1}:")
    print(f"  Title: {row['title']}")
    print(f"  Start Time: {pd.to_datetime(row['startTime'], unit='s')}")
    print(f"  End Time: {pd.to_datetime(row['endTime'], unit='s')}")
    print(f"  Location: {row['location']}")
    print(f"  Confidence: {all_predictions[top_indices[i]] * 100:.2f}%")
    print("-" * 40)
