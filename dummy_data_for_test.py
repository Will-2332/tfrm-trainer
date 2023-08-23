import random
from random import randint
import numpy as np
import pandas as pd

# Set a seed for reproducibility
np.random.seed(0)

# Define the number of samples
num_samples = 1000

# Define plausible event titles and corresponding locations with weights for realistic distribution
event_titles_locations = {
    'Meeting': (['Office', 'Library', 'Conference Room'], 5),
    'Lunch': (['Cafeteria', 'Office', 'Home', 'Restaurant'], 10),
    'Project Work': (['Office', 'Library', 'Home', 'Workshop'], 7),
    'Dinner': (['Cafeteria', 'Home', 'Restaurant'], 10),
    'Study': (['Library', 'Office', 'Home', 'Study Room'], 6),
    'Shopping': (['Market', 'Mall', 'Supermarket'], 3),
    'Reading': (['Home', 'Park', 'Library', 'Bookstore'], 4),
    'Watching TV': (['Home', 'Lounge'], 8),
    'Playing Games': (['Home', 'Park', 'Arcade'], 2),
    'Exercising': (['Gym', 'Park', 'Home', 'Sports Center'], 4),
    'Sleeping': (['Home', 'Hotel'], 10),
    'Napping': (['Home', 'Lounge'], 2),
    'Work': (['Office', 'Home', 'Library', 'Workshop'], 9),
    'Traveling': (['Airport', 'Train Station', 'Bus Station'], 3),
    'Cooking': (['Home', 'Cooking Class'], 4),
    'Hiking': (['Park', 'Mountain', 'Trail'], 2),
    'Cinema': (['Movie Theater'], 3),
    'Music': (['Concert Hall', 'Home', 'Park'], 3)
}

# Generate random titles and corresponding locations based on weights
titles = []
locations = []
grades = []
durations = []
start_hours = []
start_minutes = []
suggestions = []

# Generate weighted list of titles
weighted_titles = [title for title, (locations, weight) in event_titles_locations.items() for _ in range(weight)]

for _ in range(num_samples):
    title = random.choice(weighted_titles)
    location = random.choice(event_titles_locations[title][0])
    titles.append(title)
    locations.append(location)
    grades.append(randint(3, 5) if title in ['Reading', 'Watching TV', 'Playing Games', 'Exercising', 'Sleeping', 'Napping', 'Music'] else 0)
    suggestions.append(1 if title in ['Reading', 'Watching TV', 'Playing Games', 'Exercising', 'Sleeping', 'Napping', 'Music'] else 0)

    # Define durations and start times based on activity
    if title == 'Sleeping':
        durations.append(pd.to_timedelta(randint(6, 9), unit='h'))
        start_hours.append(np.random.randint(21, 24))
    elif title == 'Napping':
        durations.append(pd.to_timedelta(randint(1, 2), unit='h'))
        start_hours.append(np.random.randint(14, 16))
    elif title == 'Work':
        durations.append(pd.to_timedelta(randint(6, 9), unit='h'))
        start_hours.append(np.random.randint(7, 18))
    else:
        durations.append(pd.to_timedelta(randint(1, 4), unit='h'))
        start_hours.append(np.random.randint(7, 22))
    start_minutes.append(np.random.randint(0, 60))

start_times = pd.to_timedelta(start_hours, unit='h') + pd.to_timedelta(start_minutes, unit='m')
durations = pd.to_timedelta(durations)
end_times = start_times + durations
end_times = pd.to_timedelta(end_times.total_seconds() % (24 * 60 * 60), unit='s')
start_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_samples), unit='d')
start_times_str = pd.Series(start_times).dt.components.apply(lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}", axis=1)
end_times_str = pd.Series(end_times).dt.components.apply(lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}", axis=1)
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
df.to_csv('enhanced_realistic_calendar_data.csv', index=False)

df.head()
