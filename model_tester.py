import pandas as pd
import tensorflow as tf
import numpy as np

def preprocess_data(df):
    # Preprocess datetime columns
    df["startTime"] = pd.to_datetime(df["startTime"])
    df["endTime"] = pd.to_datetime(df["endTime"])

    # Extract additional features from datetime columns before converting to timestamps
    df["start_hour"] = df["startTime"].dt.hour
    df["start_weekday"] = df["startTime"].dt.weekday
    df["end_hour"] = df["endTime"].dt.hour

    # Convert datetime to timestamps for easier processing in neural networks
    df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
    df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())

    location_vocab = sorted(df["location"].unique())
    location_encoder = tf.keras.layers.StringLookup(vocabulary=location_vocab)
    df["location"] = location_encoder(df["location"]).numpy()

    return df, location_encoder

# Load the saved model
print("Loading the saved model...")
model = tf.keras.models.load_model('TFRS_LTSM_model')

# Load the CSV dataset
print("Loading CSV file...")
df = pd.read_csv('enhanced_realistic_calendar_data.csv')
print(f"Loaded {len(df)} records from the CSV file.\n")
print("First few rows of the dataset:\n", df.head())

# Preprocess the data and get the location_encoder
print("Preprocessing the data...")
df, location_encoder = preprocess_data(df)

# Generate recommendations using the model
# Drop the 'suggestion' column before creating the test_dataset
test_dict = {name: np.array(value) for name, value in df.items() if name != "suggestion"}
test_dataset = tf.data.Dataset.from_tensor_slices(test_dict)
test_dataset = test_dataset.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Ensure the model's call method is used for prediction
def predict_on_batch(inputs):
    return model(inputs, training=False)

# Instead of just storing the recommendation percentages,
# store the entire batch of data along with the recommendation percentages.
# After training, make predictions on the entire dataset
print("Making predictions on the entire dataset...")
df_dict = {name: np.array(value) for name, value in df.items() if name != "suggestion"}
dataset_for_prediction = tf.data.Dataset.from_tensor_slices(df_dict)
dataset_for_prediction = dataset_for_prediction.batch(64).prefetch(tf.data.AUTOTUNE)

predictions = []
for batch in dataset_for_prediction:
    batch_predictions = model(batch).numpy()
    for i, pred in enumerate(batch_predictions):
        predictions.append({
            "title": batch["title"].numpy()[i].decode('utf-8'),
            "startTime": pd.to_datetime(batch["startTime"].numpy()[i], unit='s'),
            "endTime": pd.to_datetime(batch["endTime"].numpy()[i], unit='s'),
            "location": location_encoder.get_vocabulary()[batch["location"].numpy()[i]],
            "probability": pred[0]
        })

# Sort the predictions based on probability in descending order
sorted_predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)

# Display the top predictions
print("\nTop Predictions with Highest Probability of Happening:")
for i, pred in enumerate(sorted_predictions[:1000]):  # Displaying top 10 predictions
    print(f"Sample {i + 1}: Title: {pred['title']}, Start Time: {pred['startTime']}, End Time: {pred['endTime']}, Location: {pred['location']}, Probability: {pred['probability']*100:.2f}%")

print("Done!")
