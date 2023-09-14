import unittest
import pandas as pd
import tensorflow as tf
import numpy as np
from model_trainer import default_suggestions

# Step 1: Load the saved model
print("Loading the saved model...")
model = tf.keras.models.load_model('TFRS_LTSM_model')


# Step 2: Function to Preprocess the Data
def preprocess_data(df):
    print("Preprocessing the data...")
    # Convert startTime and endTime to datetime format
    df["startTime"] = pd.to_datetime(df["startTime"])
    df["endTime"] = pd.to_datetime(df["endTime"])

    # Extract additional features from datetime
    df["start_hour"] = df["startTime"].dt.hour
    df["start_weekday"] = df["startTime"].dt.weekday
    df["end_hour"] = df["endTime"].dt.hour

    # Convert datetime to timestamps
    df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
    df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())

    # Encode location as integers
    location_vocab = sorted(df["location"].unique())
    location_encoder = tf.keras.layers.StringLookup(vocabulary=location_vocab)
    df["location"] = location_encoder(df["location"]).numpy()

    return df, location_encoder


# Step 3: Function to Run Inference (Not running it yet)
def run_inference(dataset_for_prediction, model, location_encoder):
    """
    Run inference on a given dataset using a trained model.

    Parameters:
        dataset_for_prediction (tf.data.Dataset): The dataset to run inference on.
        model (tf.keras.Model): The trained model.
        location_encoder (tf.keras.layers.StringLookup): The location encoder used during preprocessing.

    Returns:
        list: A list of dictionaries containing the prediction details.
    """
    # Initialize an empty list to store the predictions
    predictions = []

    try:
        # Loop through each batch in the dataset
        for batch in dataset_for_prediction:
            # Run the model's prediction method
            batch_predictions = model.predict(batch)

            # Check if the batch predictions are empty
            if batch_predictions.size == 0:
                print("Warning: Empty batch predictions.")
                continue

            # Loop through each prediction in the batch
            for i, pred in enumerate(batch_predictions):
                # Append the prediction and other details to the list
                predictions.append({
                    "title": batch["title"].numpy()[i].decode('utf-8'),
                    "startTime": pd.to_datetime(batch["startTime"].numpy()[i], unit='s'),
                    "endTime": pd.to_datetime(batch["endTime"].numpy()[i], unit='s'),
                    "location": location_encoder.get_vocabulary()[batch["location"].numpy()[i]],
                    "probability": pred
                })

        # If the predictions list is empty after running inference
        if not predictions:
            print("Warning: No recommendations made by the model.")

    except Exception as e:
        print(f"An error occurred during inference: {e}")

    return predictions



# Preprocess the data and get the location encoder
df, location_encoder = preprocess_data(df)

# Create a dictionary from the DataFrame for testing
test_dict = {name: np.array(value) for name, value in df.items() if name != "suggestion"}

# Create a TensorFlow Dataset for testing
test_dataset = tf.data.Dataset.from_tensor_slices(test_dict)
test_dataset = test_dataset.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Create a dictionary from the DataFrame for prediction
df_dict = {name: np.array(value) for name, value in df.items() if name != "suggestion"}

# Create a TensorFlow Dataset for prediction
dataset_for_prediction = tf.data.Dataset.from_tensor_slices(df_dict)
dataset_for_prediction = dataset_for_prediction.batch(64).prefetch(tf.data.AUTOTUNE)

# Step 4: Implement the test cases
class TestInferencePipeline(tf.test.TestCase):

    def test_low_grades(self):
        print("Loading CSV file...")
        df = pd.read_csv('enhanced_realistic_calendar_data.csv')
        print(f"Loaded {len(df)} records from the CSV file.\n")

        test_df = df.copy()
        test_df['grade'] = 1  # Setting all grades to 1
        test_df['suggestion'] = 0  # Setting all suggestions to 0

        test_df, _ = preprocess_data(test_df)  # Unpack the tuple

        # Create a dictionary from the DataFrame for prediction
        test_dict = {name: np.array(value) for name, value in test_df.items() if name != "suggestion"}

        # Create a TensorFlow Dataset for prediction
        test_dataset = tf.data.Dataset.from_tensor_slices(test_dict)
        test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

        predictions = run_inference(test_dataset, model, location_encoder)
        self.assertEqual(predictions, default_suggestions)

    def test_empty_data(self):
        print("Loading CSV file...")
        df = pd.read_csv('enhanced_realistic_calendar_data.csv')
        print(f"Loaded {len(df)} records from the CSV file.\n")
        test_df = pd.DataFrame()  # Empty DataFrame
        test_df, _ = preprocess_data(test_df)
        test_dataset = dataset_for_prediction(test_df)
        predictions = run_inference(test_dataset, model)
        self.assertEqual(predictions, [])  # Assuming the model returns an empty list for empty data

    def test_model_confidence(self):
        print("Loading CSV file...")
        df = pd.read_csv('enhanced_realistic_calendar_data.csv')
        print(f"Loaded {len(df)} records from the CSV file.\n")
        df['grade'] = 0  # Setting all grades to 0
        df['suggestion'] = 0
        test_df, _ = preprocess_data(df)
        test_dataset = create_tf_dataset(test_df)
        predictions = run_inference(test_dataset, model)
        self.assertEqual(predictions, default_suggestions)  # Replace default_suggestions with your actual default suggestions



if __name__ == "__main__":
    unittest.main()
import pandas as pd
import tensorflow as tf
import numpy as np
import unittest

# Step 1: Load the model
print("Loading the saved model...")
model = tf.keras.models.load_model('TFRS_LTSM_model')


# Step 2: Preprocess the data
def preprocess_data(df):
    # ... (your preprocessing code here)
    return df, location_encoder


# Step 3: Create the inference function
def run_inference(dataset, model):
    # ... (your new inference code here)
    return predictions


# Step 4: Load the data
print("Loading CSV file...")
df = pd.read_csv('enhanced_realistic_calendar_data.csv')


# Step 5: Implement the test cases
class TestInferencePipeline(tf.test.TestCase):

    def test_low_grades(self):
        test_df = df.copy()
        test_df['grade'] = 0  # Setting all grades to 0
        test_df, _ = preprocess_data(test_df)
        test_dataset = create_tf_dataset(test_df)
        predictions = run_inference(test_dataset, model)
        self.assertTrue(...)  # Your assertion here

    def test_empty_data(self):
        test_df = pd.DataFrame()  # Empty DataFrame
        test_df, _ = preprocess_data(test_df)
        test_dataset = create_tf_dataset(test_df)
        predictions = run_inference(test_dataset, model)
        self.assertTrue(...)  # Your assertion here

    def test_model_confidence(self):
        test_df = df.copy()
        test_df, _ = preprocess_data(test_df)
        test_dataset = create_tf_dataset(test_df)
        predictions = run_inference(test_dataset, model)
        self.assertTrue(...)  # Your assertion here


if __name__ == "__main__":
    unittest.main()