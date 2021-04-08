"""
Script to get all shots taken in a competition and plot their location and outcome
"""
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('competition_name', 'FIFA World Cup', 'Name of competition to plot for')

def get_competition_and_season_id(competition_name):
  competition_name = competition_name.lower()
  if competition_name == "epl":
    competition_id = 2
    season_id = 44
  elif competition_name == "laliga":
    competition_id = 11
    season_id = 42
  elif competition_name == "fifa world cup":
    competition_id = 43
    season_id = 3
  else:
    raise Exception(f'Competition {competition_name} not supported.')
  return (competition_id, season_id)

def check_shot(d):
  ID_OF_SHOT = 16
  if 'id' in d:
    return d['id'] == ID_OF_SHOT
  return False

def form_intervals(kmeans_centers):
  intervals = []
  medians = []
  for center_one, center_two in zip(kmeans_centers, kmeans_centers[1:]):
    medians.append((center_one + center_two) // 2)
  intervals.append((0, medians[0]))
  for i in range(len(medians)):
    intervals.append((medians[i], medians[i+1]))
  intervals.append((medians[-1], 93))

  return intervals

def plot_pitch_line(x_start, x_end, y_start, y_end):
  x, y = np.mgrid[x_start:x_end+1:1, y_start:y_end+1:1]
  x = np.reshape(x, -1)
  y = np.reshape(y, -1)
  plt.plot(x, y, color='black')

def main(argv):
  competition_id, season_id = get_competition_and_season_id(FLAGS.competition_name)
  matches = pd.read_json(f'../data/matches/{competition_id}/{season_id}.json')
  match_ids = matches['match_id'].to_dict().values()
  
  events_in_matches = []
  for match_id in match_ids:
    events_in_match = pd.read_json(f'../data/events/{match_id}.json')
    events_in_matches.append(events_in_match)
  events_in_matches = pd.concat(events_in_matches, axis=0)
  shots_in_matches = events_in_matches[events_in_matches['type'].apply(check_shot)]

  shot_locations_in_matches = shots_in_matches['location'].to_list()
  shot_locations_in_matches = np.array(shot_locations_in_matches).reshape((-1, 2))

  shot_in_matches_normalized = pd.json_normalize(shots_in_matches['shot'])
  shot_in_matches_outcomes = shot_in_matches_normalized['outcome.name']
  goal_shot_in_matches = shot_locations_in_matches[shot_in_matches_outcomes.isin(["Goal"])]
  no_goal_on_target_shot_in_matches = shot_locations_in_matches[shot_in_matches_outcomes.isin(["Saved", "Saved To Post"])]
  off_target_shot_in_matches = shot_locations_in_matches[~shot_in_matches_outcomes.isin(["Goal", "Saved", "Saved To Post"])]

  # plot pitch: pitch outline
  plot_pitch_line(60, 120, 0, 0)
  plot_pitch_line(60, 120, 80, 80)
  plot_pitch_line(120, 120, 0, 80)
  # plot pitch: outer box
  plot_pitch_line(102, 120, 18, 18)
  plot_pitch_line(102, 120, 62, 62)
  plot_pitch_line(102, 102, 18, 62)
  # plot pitch: inner box
  plot_pitch_line(114, 120, 30, 30)
  plot_pitch_line(114, 120, 50, 50)
  plot_pitch_line(114, 114, 30, 50)

  # off-target shots in black, shots-on-target in red, goal-shots in green
  plt.plot(off_target_shot_in_matches[:, 0], off_target_shot_in_matches[:, 1], 'o', color='black', alpha=0.7)
  plt.plot(no_goal_on_target_shot_in_matches[:, 0], no_goal_on_target_shot_in_matches[:, 1], 'o', color='red', alpha=0.7)
  plt.plot(goal_shot_in_matches[:, 0], goal_shot_in_matches[:, 1], 'o', color='green', alpha=0.7)

  # make sure y-coordinates start from top
  plt.gca().invert_yaxis()

  plt.show()

if __name__ == '__main__':
  app.run(main)