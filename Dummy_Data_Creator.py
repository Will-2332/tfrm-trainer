from concurrent.futures import ProcessPoolExecutor
import random
from random import randint
import numpy as np
import pandas as pd


def generate_data_chunk(start, end):
    np.random.seed(start)
    titles, locations, grades, durations, start_hours, start_minutes, suggestions = [], [], [], [], [], [], []

    # Define event titles, locations, and weights
    event_titles_locations = {
        'Meeting': (['Office', 'Conference Room', 'Home'], 5),
        'Lunch': (['Cafeteria', 'Home'], 10),
        'Project Work': (['Library', 'Home Office', 'Home'], 7),
        'Dinner': (['Home', 'Restaurant'], 10),
        'Study': (['Library', 'Home Office', 'Home'], 6),
        'Shopping': (['Mall', 'Grocery Store'], 3),
        'Reading': (['Park', 'Home'], 4),
        'Watching TV': (['Home'], 8),
        'Playing Games': (['Home', 'Friend\'s House'], 2),
        'Exercising': (['Gym', 'Park', 'Home'], 4),
        'Sleeping': (['Home'], 10),
        'Napping': (['Home'], 2),
        'Work': (['Office', 'Home'], 9),
        'Traveling': (['Airport', 'Train Station'], 3),
        'Cooking': (['Home'], 4),
        'Hiking': (['Park', 'Nature Reserve'], 2),
        'Cinema': (['Movie Theater'], 3),
        'Music': (['Concert Hall', 'Home'], 3),
        'Theatre': (['Theatre'], 3),
        'Museum Visit': (['Museum'], 4),
        'Date': (['Restaurant', 'Park'], 4),
        'Going Out with Friends': (['Bar', 'Park'], 6),
        'Party at Clubs': (['Club'], 3)
    }

    leisure_activities = ['Reading', 'Watching TV', 'Playing Games', 'Exercising', 'Sleeping', 'Napping', 'Music',
                          'Date', 'Going Out with Friends', 'Party at Clubs','Cinema','Theatre','Museum Visit','Hiking']

    weighted_titles = [title for title, (locations, weight) in event_titles_locations.items() for _ in range(weight)]

    start_dates_list = [(pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365), unit='d')) for _ in
                        range(start, end)]

    for _ in range(start, end):
        title = random.choice(weighted_titles)
        location = random.choice(event_titles_locations[title][0])
        titles.append(title)
        locations.append(location)
        grades.append(randint(3, 5) if title in leisure_activities else 0)
        suggestions.append(1 if title in leisure_activities else 0)

        if title == 'Work':
            while start_dates_list[_ - start].weekday() > 4:
                start_dates_list[_ - start] = start_dates_list[_ - start] + pd.to_timedelta(1, unit='d')
            durations.append(pd.to_timedelta(randint(7, 9), unit='h'))
            start_hours.append(np.random.randint(8, 10))
        elif title == 'Lunch':
            durations.append(pd.to_timedelta(randint(0, 1), unit='h'))
            start_hours.append(np.random.randint(12, 14))
        else:
            durations.append(pd.to_timedelta(randint(1, 4), unit='h'))
            start_hours.append(np.random.randint(7, 22))
        start_minutes.append(np.random.randint(0, 60))

    start_times = pd.to_timedelta(start_hours, unit='h') + pd.to_timedelta(start_minutes, unit='m')
    durations = pd.to_timedelta(durations)
    end_times = start_times + durations
    end_times = pd.to_timedelta(end_times.total_seconds() % (24 * 60 * 60), unit='s')
    start_times_str = pd.Series(start_times).dt.components.apply(
        lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}", axis=1)
    end_times_str = pd.Series(end_times).dt.components.apply(lambda x: f"{x.hours:02d}:{x.minutes:02d}:{x.seconds:02d}",
                                                             axis=1)
    start_datetimes = pd.to_datetime(pd.Series(start_dates_list).astype(str) + ' ' + start_times_str, format='%Y-%m-%d %H:%M:%S')
    end_datetimes = pd.to_datetime(pd.Series(start_dates_list).astype(str) + ' ' + end_times_str, format='%Y-%m-%d %H:%M:%S')

    return pd.DataFrame({
        'title': titles,
        'location': locations,
        'startTime': start_datetimes,
        'endTime': end_datetimes,
        'grade': grades,
        'suggestion': suggestions
    })


if __name__ == "__main__":
    num_samples = 5000
    num_workers = 16
    chunk_size = num_samples // num_workers

    futures = []
    with ProcessPoolExecutor() as executor:
        for i in range(0, num_samples, chunk_size):
            futures.append(executor.submit(generate_data_chunk, i, i + chunk_size))

    all_data_frames = [future.result() for future in futures]
    final_df = pd.concat(all_data_frames, ignore_index=True)

    final_df.to_csv('dummy_calendar_data.csv', index=False)
