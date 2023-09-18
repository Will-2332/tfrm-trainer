import pandas as pd
import tensorflow as tf
import numpy as np

print("Loading CSV file...")
df = pd.read_csv('enhanced_realistic_calendar_data.csv')
print(f"Loaded {len(df)} records from the CSV file.\n")
print("First few rows of the dataset:\n", df.head())

def preprocess_data(df):
    print("Preprocessing data...")
    df["startTime"] = pd.to_datetime(df["startTime"])
    df["endTime"] = pd.to_datetime(df["endTime"])
    df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
    df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())
    df['suggestion'] = 0
    df['grade'] = 0
    return df

print("Loading the saved model...")
model = tf.keras.models.load_model('TFRS_LTSM_model')

df = preprocess_data(df)

df_dict = {name: np.array(value) for name, value in df.items() if name != "suggestion"}
dataset_for_prediction = tf.data.Dataset.from_tensor_slices(df_dict)
dataset_for_prediction = dataset_for_prediction.batch(64).prefetch(tf.data.AUTOTUNE)

predictions = []

def run_inference(dataset_for_prediction, model):
    try:
        for batch in dataset_for_prediction:
            batch_predictions = model.predict(batch)
            if batch_predictions.size == 0:
                print("Warning: Empty batch predictions.")
                continue
            for i, pred in enumerate(batch_predictions):
                predictions.append({
                    "title": batch["title"].numpy()[i],
                    "startTime": pd.to_datetime(batch["startTime"].numpy()[i], unit='s'),
                    "endTime": pd.to_datetime(batch["endTime"].numpy()[i], unit='s'),
                    "location": batch["location"].numpy()[i],
                    "probability": pred
                })
        if not predictions:
            print("Warning: No recommendations made by the model.")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

run_inference(dataset_for_prediction, model)

if not predictions:
    print("No recommendations made by the model.")
else:
    sorted_predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
    print("\nTop Predictions with Highest Probability of Happening:")
    for i, pred in enumerate(sorted_predictions[:10]):
        print(
            f"Sample {i + 1}: Title: {pred['title']}, Start Time: {pred['startTime']}, End Time: {pred['endTime']}"
            f", Location: {pred['location']}, Probability: {pred['probability'] * 100:.2f}%")

print("Done!")
