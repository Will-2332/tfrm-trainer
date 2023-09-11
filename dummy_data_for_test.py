import random
from random import randint
import numpy as np
import pandas as pd

# Set a seed for reproducibility
np.random.seed(0)

# Define the number of samples
num_samples = 100000

# Define plausible event titles and corresponding locations with weights for realistic distribution
event_titles_locations = {
    'Meeting': (['Canary Wharf', 'The Shard', 'Barbican Centre', 'Home'], 5),
    'Lunch': (['Borough Market', 'Covent Garden', 'Brick Lane', 'Home'], 10),
    'Project Work': (['British Library', 'Kings College', 'UCL', 'Home'], 7),
    'Dinner': (['Soho', 'Shoreditch', 'Camden Town', 'Home'], 10),
    'Study': (['British Library', 'Imperial College', 'LSE', 'Home'], 6),
    'Shopping': (['Oxford Street', 'Westfield Stratford', 'King’s Road'], 3),
    'Reading': (['Hyde Park', 'Hampstead Heath', 'Southbank Centre', 'Home'], 4),
    'Watching TV': (['Home'], 8),
    'Playing Games': (['Namco Funscape', 'Four Thieves', 'Home'], 2),
    'Exercising': (['PureGym', 'The Regent’s Park', 'Virgin Active', 'Home'], 4),
    'Sleeping': (['Home'], 10),
    'Napping': (['Home'], 2),
    'Work': (['City of London', 'Tech City', 'Paddington', 'Home'], 9),
    'Traveling': (['Heathrow Airport', 'St Pancras International', 'Victoria Station'], 3),
    'Cooking': (['Home'], 4),
    'Hiking': (['Greenwich Park', 'Richmond Park'], 2),
    'Cinema': (['BFI IMAX', 'Electric Cinema'], 3),
    'Music': (['Royal Albert Hall', 'O2 Academy Brixton', 'Roundhouse'], 3),
    'Theatre': (['West End', 'National Theatre', 'The Globe'], 3),
    'Museum Visit': (['British Museum', 'Tate Modern', 'Natural History Museum'], 4),
    'Date': (['Sky Garden', 'The Shard', 'River Thames Cruise'], 4),
    'Going Out with Friends': (['Camden Market', 'Covent Garden', 'Southbank Centre'], 6),
    'Party at Clubs': (['Fabric', 'Ministry of Sound', 'Cargo'], 3)
}

leisure_activities = ['Reading', 'Watching TV', 'Playing Games', 'Exercising', 'Sleeping', 'Napping', 'Music', 'Date', 'Going Out with Friends', 'Party at Clubs']

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

start_dates_list = [(pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365), unit='d')) for _ in range(num_samples)]

for _ in range(num_samples):
    title = random.choice(weighted_titles)
    location = random.choice(event_titles_locations[title][0])
    titles.append(title)
    locations.append(location)
    grades.append(randint(3, 5) if title in leisure_activities else 0)
    suggestions.append(1 if title in leisure_activities else 0)

    # Define durations and start times based on activity
    if title == 'Work':
        # Ensure work is only on weekdays
        while start_dates_list[_].weekday() > 4:
            start_dates_list[_] = start_dates_list[_] + pd.to_timedelta(1, unit='d')
        durations.append(pd.to_timedelta(randint(7, 9), unit='h'))
        start_hours.append(np.random.randint(8, 10))
    elif title == 'Lunch':
        durations.append(pd.to_timedelta(randint(0, 1), unit='h'))
        start_hours.append(np.random.randint(12, 14))
    elif title == 'Dinner':
        durations.append(pd.to_timedelta(randint(1, 2), unit='h'))
        start_hours.append(np.random.randint(19, 21))
    elif title == 'Party at Clubs':
        durations.append(pd.to_timedelta(randint(3, 5), unit='h'))
        start_hours.append(np.random.randint(22, 24))
    elif title == 'Date':
        durations.append(pd.to_timedelta(randint(2, 4), unit='h'))
        start_hours.append(np.random.randint(19, 22))
    elif title == 'Going Out with Friends':
        durations.append(pd.to_timedelta(randint(2, 5), unit='h'))
        start_hours.append(np.random.randint(17, 20))
    elif title == 'Museum Visit':
        durations.append(pd.to_timedelta(randint(2, 4), unit='h'))
        start_hours.append(np.random.randint(10, 15))
    else:
        durations.append(pd.to_timedelta(randint(1, 4), unit='h'))
        start_hours.append(np.random.randint(7, 22))
    start_minutes.append(np.random.randint(0, 60))

start_times = pd.to_timedelta(start_hours, unit='h') + pd.to_timedelta(start_minutes, unit='m')
durations = pd.to_timedelta(durations)
end_times = start_times + durations
end_times = pd.to_timedelta(end_times.total_seconds() % (24 * 60 * 60), unit='s')
start_times_str = pd.Series(start_times).dt.components.apply(lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}", axis=1)
end_times_str = pd.Series(end_times).dt.components.apply(lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}", axis=1)
start_datetimes = pd.to_datetime(pd.Series(start_dates_list).astype(str) + ' ' + start_times_str)
end_datetimes = pd.to_datetime(pd.Series(start_dates_list).astype(str) + ' ' + end_times_str)


# Create a DataFrame
df = pd.DataFrame({
    'title': titles,
    'location': locations,
    'startTime': start_datetimes,
    'endTime': end_datetimes,
    'grade': grades,
    'suggestion': suggestions
})

# # Drop rows where the suggestion equals 1
# df = df[df['suggestion'] != 1]

# Save the DataFrame to a CSV file
df.to_csv('enhanced_realistic_calendar_data.csv', index=False)

df.head()