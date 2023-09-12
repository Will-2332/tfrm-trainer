import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from sklearn.model_selection import train_test_split

# Check the version of TensorFlow and available GPUs
print("TensorFlow Version: ", tf.__version__)
print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

# Load the data
print("Loading data...")
df = pd.read_csv('dummy_calendar_data.csv')

# Display basic information about the loaded data
print("Total number of samples:", len(df))
print("Missing values in each column:\n", df.isnull().sum())
print("First few rows of the DataFrame:\n", df.head())

# Preprocess datetime columns
print("Preprocessing data...")
df["startTime"] = pd.to_datetime(df["startTime"])
df["endTime"] = pd.to_datetime(df["endTime"])

# Extract additional features from datetime columns before converting to timestamps
df["start_hour"] = df["startTime"].dt.hour
df["start_weekday"] = df["startTime"].dt.weekday
df["end_hour"] = df["endTime"].dt.hour

# Convert datetime to timestamps for easier processing in neural networks
df['startTime'] = df['startTime'].apply(lambda x: x.timestamp())
df['endTime'] = df['endTime'].apply(lambda x: x.timestamp())

# Get the unique activity titles
unique_activity_titles = df["title"].unique().tolist()

location_vocab = sorted(df["location"].unique())
location_encoder = tf.keras.layers.StringLookup(vocabulary=location_vocab)
df["location"] = location_encoder(df["location"]).numpy()

# Dataset with only suggestions for the model to use as its default
df2 = df[df['suggestion'] != 0]
default_suggestions = df2['suggestion'].unique().tolist()

# Split the data into training and testing sets
print("Splitting data into train and test sets...")
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Convert pandas DataFrames to TensorFlow Datasets
print("Converting data to TensorFlow Datasets...")
train_dict = {name: np.array(value) for name, value in train.items()}
test_dict = {name: np.array(value) for name, value in test.items()}

train = tf.data.Dataset.from_tensor_slices((train_dict, train_dict.pop("suggestion")))
test = tf.data.Dataset.from_tensor_slices((test_dict, test_dict.pop("suggestion")))

# Display shapes of a few samples from the datasets
def print_shapes(dataset, num_samples=5):
    for i, (features, label) in enumerate(dataset.take(num_samples)):
        print(f"Sample {i + 1} shapes:")
        for key, value in features.items():
            print(f"{key}: {value.shape}")

print_shapes(train)
print_shapes(test)

# Shuffle, batch, and prefetch the datasets for better performance during training
train = train.shuffle(buffer_size=len(train))
train = train.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)  # Changed batch size to 64
test = test.batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)  # Changed batch size to 64

# Cache the datasets to speed up subsequent epochs
print("Preparing the data for training...")
cached_train = train.cache()
cached_test = test.cache()

# Display batch sizes
def print_batch_sizes(dataset):
    batch_sizes = [batch[0]['title'].shape[0] for batch in dataset]
    print("Batch sizes:", batch_sizes)

print_batch_sizes(cached_train)
print_batch_sizes(cached_test)

# Check that the dataset isn't empty
print("Number of batches in train dataset:", len(list(cached_train)))

# Normalize 'grade', 'startTime', and 'endTime' features
print("Normalizing features...")
grade_normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
grade_normalizer.adapt(train.map(lambda x, _: tf.reshape(x["grade"], [-1, 1])))

start_time_normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
start_time_normalizer.adapt(train.map(lambda x, _: tf.reshape(x["startTime"], [-1, 1])))

end_time_normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
end_time_normalizer.adapt(train.map(lambda x, _: tf.reshape(x["endTime"], [-1, 1])))

# Define user and activity models
print("Defining user and activity models...")

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.unique_activity_titles = unique_activity_titles
        self.title_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token='')
        self.title_lookup.adapt(unique_activity_titles)
        self.title_embedding = tf.keras.layers.Embedding(len(unique_activity_titles) + 2, 32)
        self.startTime_embedding = tf.keras.layers.Embedding(input_dim=24, output_dim=16)  # 24 hours in a day
        self.endTime_embedding = tf.keras.layers.Embedding(input_dim=24, output_dim=16)  # 24 hours in a day
        self.flatten = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(64)  # Add LSTM layer

    def get_config(self):
        return {"unique_activity_titles": self.unique_activity_titles}  # Add any arguments that your __init__ method uses here

    def call(self, inputs):
        title_indices = self.title_lookup(inputs["title"])
        startTime_indices = tf.cast(inputs["startTime"] % (60*60*24) // (60*60), tf.int32)  # Extract hour of day
        endTime_indices = tf.cast(inputs["endTime"] % (60*60*24) // (60*60), tf.int32)  # Extract hour of day
        embeddings = tf.concat([
            self.title_embedding(title_indices),
            self.startTime_embedding(startTime_indices),
            self.endTime_embedding(endTime_indices),
        ], axis=-1)
        embeddings = tf.expand_dims(embeddings, 1)
        lstm_output = self.lstm(embeddings)
        return self.flatten(lstm_output)


class ActivityModel(tf.keras.Model):
    def __init__(self, grade_normalizer, unique_activity_titles):
        super().__init__()
        self.grade_normalizer = grade_normalizer  # Add this line
        self.unique_activity_titles = unique_activity_titles  # Add this line
        self.title_lookup = tf.keras.layers.StringLookup(vocabulary=unique_activity_titles, mask_token=None)
        self.title_embedding = tf.keras.layers.Embedding(len(unique_activity_titles) + 1, 32)
        self.grade_processing = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])

    def get_config(self):
        return {"grade_normalizer": self.grade_normalizer, "unique_activity_titles": self.unique_activity_titles}

    def call(self, inputs):
        title = self.title_lookup(inputs["title"])
        title_embed = self.title_embedding(title)
        grade_embed = self.grade_processing(tf.expand_dims(inputs["grade"], axis=-1))
        return tf.concat([title_embed, grade_embed], axis=-1)



class ActivityRecommenderModel(tfrs.models.Model):
    def __init__(self, default_suggestions, unique_activity_titles, confidence_threshold=0.75):
        super().__init__()
        self.user_model = UserModel()
        self.activity_model = ActivityModel(grade_normalizer, unique_activity_titles)
        self.default_suggestions = tf.constant(default_suggestions, dtype=tf.float32)
        self.confidence_threshold = confidence_threshold
        self.task = tfrs.tasks.Ranking(
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    def get_config(self):
        return {"default_suggestions": self.default_suggestions.numpy().tolist(),
                "confidence_threshold": self.confidence_threshold}

    def call(self, inputs, training=True):
        user_features = {name: inputs[name] for name in ["title", "startTime", "endTime"]}
        activity_features = {name: inputs[name] for name in ["title", "grade"]}
        user_embeddings = self.user_model(user_features)[:, None]
        activity_embeddings = self.activity_model(activity_features)[:, None]
        recommendations = tf.sigmoid(tf.reduce_sum(user_embeddings * activity_embeddings, axis=-1))

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
        features, targets = inputs
        predictions = self(features, training=training)
        return self.task(targets, predictions)



# Instantiate and compile the model
print("Compiling the model...")
model = ActivityRecommenderModel(default_suggestions,unique_activity_titles)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))  # Changed learning rate to 1e-4

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# Train the model
print("Starting training...")
history = model.fit(cached_train, epochs=10, validation_data=cached_test, callbacks=[early_stopping])  # Changed epochs to 10

# Evaluate the model
print("Evaluating the model...")
evaluation_results = model.evaluate(cached_test, return_dict=True)

# Print evaluation results
print("Evaluation Results: ", evaluation_results)

# Save the model
print("Saving the model...")
model.save('TFRS_LTSM_model')

print("Done!")