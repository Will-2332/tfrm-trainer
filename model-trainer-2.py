import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split

# Check the version of TensorFlow and available GPUs
print("TensorFlow Version: ", tf.__version__)
print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

# Load the data
print("Loading data...")
df = pd.read_csv('dummy_calendar_data.csv')

# Show first few rows of the DataFrame
print("First few rows of the DataFrame:\n", df.head())

# Preprocessing data
print("Preprocessing data...")
df['startTime'] = pd.to_datetime(df['startTime'])
df['endTime'] = pd.to_datetime(df['endTime'])
df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())

# Splitting data into train and test sets
print("Splitting data into train and test sets...")
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Converting the train and test sets to TensorFlow Datasets
print("Converting data to TensorFlow Datasets...")
train = tf.data.Dataset.from_tensor_slices(dict(train))
test = tf.data.Dataset.from_tensor_slices(dict(test))

print(f"Training set size: {len(train)}, Test set size: {len(test)}")

# User Model
print("Defining user model...")
class UserModel(tf.keras.Model):

    def __init__(self):
        super(UserModel, self).__init__()

        # This is an embedding layer for 'title', it maps from raw input features
        # (strings in this case) to dense vectors.
        # The StringLookup layer translates the string input to integer indices,
        # and the Embedding layer maps these integer indices to dense vectors.
        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=tf.unique(df.title)[0],  # Use unique titles as vocabulary
                mask_token=None  # Do not mask any token
            ),
            tf.keras.layers.Embedding(len(tf.unique(df.title)[0]) + 1, 32)  # 32 is the size of embedding vector
        ])

    def call(self, inputs):
        # When the model is called with some inputs, it passes the inputs
        # through the embedding layer and returns the embeddings.
        print("\nUserModel inputs:", inputs)
        print("\nUserModel title_embedding:", self.title_embedding(inputs))
        return self.title_embedding(inputs)


# Activity Model
print("Defining activity model...")
unique_activity_titles = df['title'].unique().tolist()
unique_activity_locations = df['location'].unique().tolist()
startTime_bins = np.linspace(df['startTime'].min(), df['startTime'].max(), num=10)
endTime_bins = np.linspace(df['endTime'].min(), df['endTime'].max(), num=10)

class ActivityRecommendationModel(tfrs.models.Model):

    def __init__(self, user_model, activity_model, train):
        super(ActivityRecommendationModel, self).__init__()
        self.user_model = user_model
        self.activity_model = activity_model

        # The task object is constructed for retrieval task.
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=train.batch(128).map(self.activity_model),
            ),
        )

    def __init__(self, user_model, activity_model, train):
        super(ActivityRecommendationModel, self).__init__()
        self.user_model = user_model
        self.activity_model = activity_model

        # The task object is constructed for retrieval task.
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=train.batch(128).map(self.activity_model),
            ),
        )

    def call(self, features):
        user_embeddings = self.user_model(features["title"])
        positive_activity_embeddings = self.activity_model(features)
        return user_embeddings, positive_activity_embeddings

    def compute_loss(self, features, training=False):
        user_embeddings, positive_activity_embeddings = self.call(features)
        return self.task(user_embeddings, positive_activity_embeddings)



    def compute_output_signature(self, input_signature):
        return tf.TensorSpec(shape=(input_signature[0].shape[0], 97), dtype=tf.float32)


    # Take one batch of the training data
    sample_batch = next(iter(train.batch(1)))

    # Extract a batch of 'title'
    title_batch = {'title': sample_batch['title']}

    # Extract a batch of 'location', 'startTime', 'endTime', and 'grade'
    activity_batch = sample_batch

# Create instances of UserModel and ActivityModel
user_model = UserModel()
activity_model = ActivityModel()

# Take one batch of the training data
sample_batch = next(iter(train.batch(1)))

# Extract a batch of 'title'
title_batch = {'title': sample_batch['title']}

# Extract a batch of 'location', 'startTime', 'endTime', and 'grade'
activity_batch = sample_batch

# Call the models with the sample batches and print the output
print("Output of UserModel:", user_model(title_batch['title']))

print("Output of ActivityModel:", activity_model(activity_batch))

# Create an instance of ActivityRecommendationModel
activity_recommendation_model = ActivityRecommendationModel(user_model, activity_model, train.batch(128))

# Print the output of the model
print("Output of ActivityRecommendationModel:", activity_recommendation_model(sample_batch))