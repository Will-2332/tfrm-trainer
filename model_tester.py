import pandas as pd
import tensorflow as tf
import numpy as np

def preprocess_data(df):
    df["startTime"] = pd.to_datetime(df["startTime"])
    df["endTime"] = pd.to_datetime(df["endTime"])
    df["start_hour"] = df["startTime"].dt.hour
    df["start_weekday"] = df["startTime"].dt.weekday
    df["end_hour"] = df["endTime"].dt.hour
    df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
    df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())
    location_vocab = sorted(df["location"].unique())
    location_encoder = tf.keras.layers.StringLookup(vocabulary=location_vocab)
    df["location"] = location_encoder(df["location"]).numpy()
    return df, location_encoder

def run_inference(dataset_for_prediction, model, location_encoder):
    predictions = []
    for batch in dataset_for_prediction:
        batch_predictions = model(batch, training=False).numpy()
        if batch_predictions.size == 0:
            print("Warning: Empty batch predictions.")
            continue
        for i, pred in enumerate(batch_predictions):
            predictions.append({
                "title": batch["title"].numpy()[i].decode('utf-8'),
                "startTime": pd.to_datetime(batch["startTime"].numpy()[i], unit='s'),
                "endTime": pd.to_datetime(batch["endTime"].numpy()[i], unit='s'),
                "location": location_encoder.get_vocabulary()[batch["location"].numpy()[i]],
                "probability": pred
            })
    return predictions

print("Loading the saved model...")
model = tf.keras.models.load_model('TFRS_LTSM_model')

print("Loading CSV file...")
df = pd.read_csv('enhanced_realistic_calendar_data.csv')
print(f"Loaded {len(df)} records from the CSV file.\n")
print("First few rows of the dataset:\n", df.head())

print("Preprocessing the data...")
df, location_encoder = preprocess_data(df)

test_dict = {name: np.array(value) for name, value in df.items() if name != "suggestion"}
test_dataset = tf.data.Dataset.from_tensor_slices(test_dict)
test_dataset = test_dataset.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

df_dict = {name: np.array(value) for name, value in df.items() if name != "suggestion"}
dataset_for_prediction = tf.data.Dataset.from_tensor_slices(df_dict)
dataset_for_prediction = dataset_for_prediction.batch(64).prefetch(tf.data.AUTOTUNE)

predictions = run_inference(dataset_for_prediction, model, location_encoder)

if not predictions:
    print("No recommendations made by the model.")
else:
    sorted_predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
    print("\nTop Predictions with Highest Probability of Happening:")
    for i, pred in enumerate(sorted_predictions[:10]):
        print(f"Sample {i + 1}: Title: {pred['title']}, Start Time: {pred['startTime']}, End Time: {pred['endTime']}, Location: {pred['location']}, Probability: {pred['probability']*100:.2f}%")

print("Done!")
