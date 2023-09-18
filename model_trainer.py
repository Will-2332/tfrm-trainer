import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow_recommenders.metrics import FactorizedTopK

# Check TensorFlow version and available GPUs
print("TensorFlow Version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Load the data
print("Loading data...")
df = pd.read_csv('dummy_calendar_data.csv')

# Display basic information about the loaded data
print(f"Total number of samples: {len(df)}")
print(f"Missing values in each column:\n{df.isnull().sum()}")
print(f"First few rows of the DataFrame:\n{df.head()}")

# Preprocess datetime columns
print("Preprocessing data...")
df["startTime"] = pd.to_datetime(df["startTime"])
df["endTime"] = pd.to_datetime(df["endTime"])

# Convert datetime to timestamps
df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())

# Get unique activity titles
unique_activity_titles = df["title"].unique().tolist()

# Get unique locations
unique_locations = df["location"].unique().tolist()

# Dataset with only suggestions for the model to use as its default
df2 = df[df['suggestion'] != 0]
default_suggestions = df2['suggestion'].unique().tolist()

# Filter the datasets to only include suggestions
df_user_filtered = df[df['suggestion'] == 1]

# Split the data into training and testing sets
print("Splitting data into train, train_user, test and test_user sets...")
train_user, test_user = train_test_split(df_user_filtered, test_size=0.2, random_state=42)
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Convert pandas DataFrames to TensorFlow Datasets
print("Converting data to TensorFlow Datasets...")
train_dict = {name: np.array(value) for name, value in train.items()}
test_dict = {name: np.array(value) for name, value in test.items()}

train_user_dict = {name: np.array(value) for name, value in train_user.items()}
test_user_dict = {name: np.array(value) for name, value in test_user.items()}

train_user = tf.data.Dataset.from_tensor_slices((train_user_dict, train_user_dict.pop("suggestion")))
test_user = tf.data.Dataset.from_tensor_slices((test_user_dict, test_user_dict.pop("suggestion")))

train = tf.data.Dataset.from_tensor_slices((train_dict, train_dict.pop("suggestion")))
test = tf.data.Dataset.from_tensor_slices((test_dict, test_dict.pop("suggestion")))

# Display shapes of a single sample
print("Displaying shapes of a single sample each...")

for features, label in train.take(1):
    print("Sample shapes train:")
    for key, value in features.items():
        print(f"{key}: {value.shape}")

for features, label in test.take(1):
    print("Sample shapes test:")
    for key, value in features.items():
        print(f"{key}: {value.shape}")

for features, label in train_user.take(1):
    print("Sample shapes train_user:")
    for key, value in features.items():
        print(f"{key}: {value.shape}")

for features, label in test_user.take(1):
    print("Sample shapes test_user:")
    for key, value in features.items():
        print(f"{key}: {value.shape}")

# For the main training set
train = train.shuffle(buffer_size=len(train))
train = train.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# For the main test set
test = test.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# For the user-specific training set
train_user = train_user.shuffle(buffer_size=len(train_user))
train_user = train_user.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# For the user-specific test set
test_user = test_user.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
# Cache datasets
print("Caching datasets...")
cached_train = train.cache()
cached_test = test.cache()
cached_train_user = train_user.cache()
cached_test_user = test_user.cache()

# Display batch sizes
print("Displaying batch sizes...")

batch_sizes_train = [batch[0]['title'].shape[0] for batch in cached_train_user]
print("Batch sizes for UserModel train:", batch_sizes_train)

batch_sizes_test = [batch[0]['title'].shape[0] for batch in cached_test_user]
print("Batch sizes for UserModel test:", batch_sizes_test)

batch_sizes_train = [batch[0]['title'].shape[0] for batch in cached_train]
print("Batch sizes for ActivityModel train:", batch_sizes_train)

batch_sizes_test = [batch[0]['title'].shape[0] for batch in cached_test]
print("Batch sizes for ActivityModel test:", batch_sizes_test)

# Check dataset size
print(f"Number of batches in UserModel train dataset: {len(list(cached_train_user))}")
print(f"Number of batches in UserModel test dataset: {len(list(cached_test_user))}")
print(f"Number of batches in ActivityModel train dataset: {len(list(cached_train))}")
print(f"Number of batches in ActivityModel test dataset: {len(list(cached_test))}")

# Define user and activity models
print("Defining user and activity models...")


class UserModel(tf.keras.Model):
    def __init__(self):
        print("Initializing UserModel...")
        super().__init__()
        self.unique_activity_titles = unique_activity_titles
        self.title_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_activity_titles,
                                                                                    mask_token=None,
                                                                                    oov_token="UNKNOWN")
        self.title_embedding = tf.keras.layers.Embedding(len(unique_activity_titles) + 2, 32)
        self.unique_location = unique_locations
        self.location_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_locations,
                                                                                       mask_token=None,
                                                                                       oov_token="UNKNOWN")
        self.location_embedding = tf.keras.layers.Embedding(len(unique_locations) + 2, 32)
        self.startTime_embedding = tf.keras.layers.Embedding(input_dim=24, output_dim=16)
        self.endTime_embedding = tf.keras.layers.Embedding(input_dim=24, output_dim=16)
        self.flatten = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)

    def get_config(self):
        return {
            "unique_activity_titles": self.unique_activity_titles,
            "unique_locations": self.unique_location
        }

    def call(self, inputs):
        title_indices = self.title_lookup(inputs["title"])
        startTime_indices = tf.cast(inputs["startTime"] % (60 * 60 * 24) // (60 * 60), tf.int32)
        endTime_indices = tf.cast(inputs["endTime"] % (60 * 60 * 24) // (60 * 60), tf.int32)
        location_indices = self.location_lookup(inputs["location"])
        location_embeddings = self.location_embedding(location_indices)

        embeddings = tf.concat([
            self.title_embedding(title_indices),
            self.startTime_embedding(startTime_indices),
            self.endTime_embedding(endTime_indices),
            location_embeddings,
        ], axis=-1)

        # Sort by startTime
        sorted_indices = tf.argsort(startTime_indices, axis=-1, direction='ASCENDING')
        sorted_embeddings = tf.gather(embeddings, sorted_indices)

        sorted_embeddings = tf.expand_dims(sorted_embeddings, 1)

        # Pass through LSTM
        lstm_output = self.lstm(sorted_embeddings)

        return self.flatten(lstm_output)


class ActivityModel(tf.keras.Model):
    def __init__(self):
        print("Initializing ActivityModel...")
        super().__init__()
        self.unique_activity_titles = unique_activity_titles
        self.title_lookup = tf.keras.layers.StringLookup(vocabulary=unique_activity_titles, mask_token=None,
                                                         oov_token="UNKNOWN")
        self.title_embedding = tf.keras.layers.Embedding(len(unique_activity_titles) + 1, 32)
        self.unique_location = unique_locations
        self.location_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_locations,
                                                                                       mask_token=None,
                                                                                       oov_token="UNKNOWN")
        self.location_embedding = tf.keras.layers.Embedding(len(unique_locations) + 2, 31)

    def get_config(self):
        return {
            "unique_activity_titles": self.unique_activity_titles,
            "unique_locations": self.unique_location
        }

    def call(self, inputs):
        title = self.title_lookup(inputs["title"])
        title_embed = self.title_embedding(title)
        grade_embed = tf.expand_dims(inputs["grade"], axis=-1)
        grade_embed = tf.cast(grade_embed, dtype=tf.float32)
        location_indices = self.location_lookup(inputs["location"])
        location_embeddings = self.location_embedding(location_indices)

        embeddings = tf.concat([title_embed, grade_embed, location_embeddings], axis=-1)

        return embeddings


# ActivityRecommenderModel now accepts pre-trained UserModel and ActivityModel
class ActivityRecommenderModel(tfrs.models.Model):

    def __init__(self, user_model, activity_model, default_suggestions, confidence_threshold=0.75):
        print("Initializing ActivityRecommenderModel...")
        super().__init__()
        self.user_model = user_model  # Now accepts a pre-trained UserModel
        self.activity_model = activity_model  # Now accepts a pre-trained ActivityModel
        self.default_suggestions = tf.constant(default_suggestions, dtype=tf.float32)
        self.confidence_threshold = confidence_threshold
        self.attention_layer = tf.keras.layers.Attention(use_scale=True)
        self.dense_layer = tf.keras.layers.Dense(64, activation='relu')
        self.task = tfrs.tasks.Ranking(
            metrics=[tf.keras.metrics.MeanAbsoluteError()
                     # ,FactorizedTopK(candidates=train_user.batch(128).map(lambda x, y: activity_model(x)))
                     ]
        )

    def get_config(self):
        return {"default_suggestions": self.default_suggestions.numpy().tolist(),
                "confidence_threshold": self.confidence_threshold}

    def call(self, inputs, training=True):
        # Common features for both user and activity models
        common_features = {
            "title": inputs["title"],
            "startTime": inputs["startTime"],
            "endTime": inputs["endTime"],
            "location": inputs["location"],
            "grade": inputs["grade"]
        }

        user_embeddings = self.user_model(common_features)[:, None]
        activity_embeddings = self.activity_model(common_features)[:, None]
        attention = self.attention_layer([user_embeddings, activity_embeddings])
        recommendations = self.dense_layer(attention)

        if not training:  # Only apply this logic during inference
            # Check if all grades are below 3
            all_grades_below_3 = tf.reduce_all(inputs['grade'] < 3)

            # Check if dataset is empty
            is_dataset_empty = tf.equal(tf.size(inputs['grade']), 0)

            # Calculate mean recommendations
            mean_recommendations = tf.reduce_mean(recommendations)

            def true_fn():
                return self.default_suggestions

            def false_fn():
                return tf.cond(
                    mean_recommendations < self.confidence_threshold,
                    lambda: self.default_suggestions,
                    lambda: recommendations
                )

            # Apply conditions
            final_recommendations = tf.case([
                (all_grades_below_3, true_fn),
                (is_dataset_empty, true_fn),
            ], default=false_fn)

            return final_recommendations

        else:
            return recommendations  # Existing logic during training

    def compute_loss(self, inputs, training=True):
        print("Computing loss...")
        features, targets = inputs
        predictions = self(features, training=training)
        return self.task(targets, predictions)


# Instantiate UserModel and ActivityModel
user_model = UserModel()
activity_model = ActivityModel()

# Compile and train UserModel
print("Compiling the UserModel...")
user_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=tf.keras.losses.MeanSquaredError())

print("Training UserModel...")
user_model.fit(train_user, epochs=10, validation_data=test_user)
print("UserModel training complete.")

# Compile and train ActivityModel
print("Compiling the ActivityModel...")
activity_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss=tf.keras.losses.MeanSquaredError())

print("Training ActivityModel...")
activity_model.fit(train, epochs=10, validation_data=test)
print("ActivityModel training complete.")

# Instantiate and compile the ActivityRecommenderModel
print("Compiling the ActivityRecommenderModel...")
model = ActivityRecommenderModel(user_model, activity_model, default_suggestions, confidence_threshold=0.75)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
print("ActivityRecommenderModel compiled.")

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# Train the model
print("Starting training of ActivityRecommenderModel...")
history = model.fit(cached_train, epochs=10, validation_data=cached_test, callbacks=[early_stopping])
print("Training of ActivityRecommenderModel complete.")

# Evaluate the model
print("Evaluating the ActivityRecommenderModel...")
evaluation_results = model.evaluate(cached_test, return_dict=True)
print("Evaluation complete.")

# Print evaluation results
print(f"Evaluation Results: {evaluation_results}")

# Save the model
print("Saving the model...")
model.save('TFRS_LTSM_model', save_format='tf', save_traces=True)
print("Model saved. Done!")

print("Loading the saved model...")
loaded_model = tf.keras.models.load_model('TFRS_LTSM_model')
print("Model loaded successfully.")
