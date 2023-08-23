import random
from random import randint

import numpy as np
import pandas as pd

# Set a seed for reproducibility
np.random.seed(0)

# Define the number of samples
num_samples = 100000

# Define plausible event titles and corresponding locations
event_titles_locations = {
    'Meeting': ['Office', 'Library'],
    'Lunch': ['Cafeteria', 'Office', 'Home'],
    'Project Work': ['Office', 'Library', 'Home'],
    'Dinner': ['Cafeteria', 'Home'],
    'Study': ['Library', 'Office', 'Home'],
    'Shopping': ['Market'],
    'Reading': ['Home', 'Park', 'Library'],
    'Watching TV': ['Home'],
    'Playing Games': ['Home', 'Park'],
    'Exercising': ['Gym', 'Park', 'Home'],
    'Sleeping': ['Home'],
    'Napping': ['Home'],
    'Work': ['Office', 'Home', 'Library']
}

# Define leisure activities
leisure_activities = ['Reading', 'Watching TV', 'Playing Games', 'Exercising', 'Sleeping', 'Napping']

# Generate random titles and corresponding locations
titles = []
locations = []
grades = []
durations = []
start_hours = []
start_minutes = []
suggestions = []

for _ in range(num_samples):
    title, location = random.choice(list(event_titles_locations.items()))
    titles.append(title)
    locations.append(random.choice(location))
    # Assign higher grades to leisure activities
    grades.append(randint(3, 5) if title in leisure_activities else 0)

    #Assign suggestions to leisure activities
    suggestions.append(1 if title in leisure_activities else 0)
    # Define special duration for Sleeping and Napping
    if title == 'Sleeping':
        durations.append(pd.to_timedelta(randint(6, 9), unit='h'))  # sleeping durations between 6 and 9 hours
        # Set start times for Sleeping between 9 PM and 12 AM
        start_hours.append(np.random.randint(21, 24))
    elif title == 'Napping':
        durations.append(pd.to_timedelta(randint(1, 2), unit='h'))  # napping durations between 1 and 2 hours
        # Set start times for Napping between 2 PM and 4 PM
        start_hours.append(np.random.randint(14, 16))
    elif title == 'Work':
        durations.append(pd.to_timedelta(randint(6, 9), unit='h'))  # Work durations between 6 and 9 hour
        start_hours.append(np.random.randint(7, 18))  # Set start times for work between 7 AM and 6 PM
    else:
        durations.append(pd.to_timedelta(randint(1, 4), unit='h'))  # durations between 1 and 3 hours for other activities
        # Set start times for other activities within typical waking hours
        start_hours.append(np.random.randint(7, 22))
    start_minutes.append(np.random.randint(0, 60))

start_times = pd.to_timedelta(start_hours, unit='h') + pd.to_timedelta(start_minutes, unit='m')

# Convert durations list to TimedeltaIndex
durations = pd.to_timedelta(durations)

# Calculate end times
end_times = start_times + durations

# Ensure end times don't go over midnight
end_times = pd.to_timedelta(end_times.total_seconds() % (24 * 60 * 60), unit='s')

# Generate random dates within a year
start_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_samples), unit='d')

# Convert times to 'HH:MM:SS' format
start_times_str = pd.Series(start_times).dt.components.apply(lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}", axis=1)
end_times_str = pd.Series(end_times).dt.components.apply(lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}", axis=1)

# Combine dates and times
start_datetimes = pd.to_datetime(pd.Series(start_dates).dt.date.astype(str) + ' ' + start_times_str)
end_datetimes = pd.to_datetime(pd.Series(start_dates).dt.date.astype(str) + ' ' + end_times_str)


# Create a DataFrame
df = pd.DataFrame({
    'title': titles,
    'location': locations,
    'startTime': start_datetimes,
    'endTime': end_datetimes,
    'grade': grades,
    'suggestion': suggestions
})

# Save the DataFrame to a CSV file
df.to_csv('dummy_calendar_data.csv', index=False)

df.head()
