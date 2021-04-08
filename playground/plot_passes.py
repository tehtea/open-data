"""
Script to get all passes made in a competition. Then, plot a random subset of these passes
"""
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from absl import app
from absl import flags

from socplot.pitch import Pitch

pitch = Pitch()
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

def check_pass(d):
  ID_OF_PASS = 30
  if 'id' in d:
    return d['id'] == ID_OF_PASS
  return False

def main(argv):
  competition_id, season_id = get_competition_and_season_id(FLAGS.competition_name)
  matches = pd.read_json(f'../data/matches/{competition_id}/{season_id}.json')
  match_ids = matches['match_id'].to_dict().values()
  
  events_in_matches = []
  for match_id in match_ids:
    events_in_match = pd.read_json(f'../data/events/{match_id}.json')
    events_in_matches.append(events_in_match)
  events_in_matches = pd.concat(events_in_matches, axis=0)
  passes_in_matches = events_in_matches[events_in_matches['type'].apply(check_pass)]
  for pass_in_match in np.random.choice(passes_in_matches.to_dict('record'), 500):
    pass_start_loc = pass_in_match['location']
    pass_end_loc = pass_in_match['pass']['end_location']
    pass_successful = not ('outcome' in pass_in_match['pass'])
    pitch.plot_pass(start=pass_start_loc, end=pass_end_loc, color='green' if pass_successful else 'red')

  plt.show()

if __name__ == '__main__':
  app.run(main)