"""
Script to get all shots and passes taken/faced in a competition by a team,
then generates some probability values.
"""
from pprint import pprint
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('competition_name', 'FIFA World Cup', 'Name of competition the team is in.')
flags.DEFINE_string('team_name', 'England', 'Team name that you want to look up.')
flags.DEFINE_string('place', 'both', 'Whether to look up stats for home, away or both.')
flags.DEFINE_string('team_prefix', None, 'Prefix to set for probability output. Can be \'teamOne\' or \'teamTwo\'. Default is None')

def weird_division(n, d):
  """
  Utility for safe divide by zero. Source: https://stackoverflow.com/questions/27317517/make-division-by-zero-equal-to-zero
  """
  return n / d if d else 0

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
  
def is_team_and_place(matches):
  def is_team(d):
      if FLAGS.place.lower() == 'home':
        return d['home_team_name'].lower() == FLAGS.team_name.lower()
      elif FLAGS.place.lower() == 'away':
        return d['away_team_name'].lower() == FLAGS.team_name.lower()
      elif FLAGS.place.lower() == 'both':
        return ('home_team_name' in d and d['home_team_name'].lower() == FLAGS.team_name.lower()) or \
          ('away_team_name' in d and d['away_team_name'].lower() == FLAGS.team_name.lower())
      return False
  if FLAGS.place.lower() == 'home':
    matches = matches[matches['home_team'].apply(is_team)]
  elif FLAGS.place.lower() == 'away':
    matches = matches[matches['away_team'].apply(is_team)]
  elif FLAGS.place.lower() == 'both':
    matches = matches[matches['home_team'].apply(is_team) | matches['away_team'].apply(is_team)]
  return matches

def check_shot(d):
  ID_OF_SHOT = 16
  if 'id' in d:
    return d['id'] == ID_OF_SHOT
  return False

def check_pass(d):
  ID_OF_PASS = 30
  if 'id' in d:
    return d['id'] == ID_OF_PASS
  return False

def check_is_current_team(d):
  if 'name' in d:
    return d['name'].lower() == FLAGS.team_name.lower()
  return False

def _get_zone(location):
  # xmin, ymin, xmax, ymax
  # Zone 1: (90, 0, 120, 30)
  # Zone 2: (114, 30, 120, 50)
  # Zone 3: (90, 50, 120, 80)
  # Zone 4: (102, 20, 114, 60)
  # Zone 5: (90, 20, 102, 60)
  x, y = location
  if 90 < x < 120 and 0 < y < 30:
    return 1
  elif 114 < x < 120 and 30 < y < 50:
    return 2
  elif 90 < x < 120 and 50 < y < 80:
    return 3
  elif 102 < x < 114 and 20 < y < 60:
    return 4
  elif 90 < x < 102 and 20 < y < 60:
    return 5
  else:
    return -1
    
def get_zone_shot_stats(shots):
  shots['shot_zone'] = shots.apply(lambda row: _get_zone(row['location']), axis=1)
  zone_stats = [(i, {'on_target': 0, 'goal': 0, 'off_target': 0, 'total_shots': 0}) for i in range(1, 6)]
  zone_stats = dict(zone_stats)
  for shot in shots.to_dict('records'):
    if shot['shot_zone'] == -1:
      continue
    shot_zone = shot['shot_zone']
    shot_outcome = shot['shot']['outcome']['name']
    if shot_outcome == 'Goal':
      zone_stats[shot_zone]['goal'] += 1
    if shot_outcome in ['Goal', 'Saved', 'Saved To Post']:
      zone_stats[shot_zone]['on_target'] += 1
    else:
      zone_stats[shot_zone]['off_target'] += 1
    zone_stats[shot_zone]['total_shots'] += 1
  return zone_stats

def valid_targets_per_zone(zone):
  if zone == 1:
    return [2, 4]
  elif zone == 2:
    return []
  elif zone == 3:
    return [2, 4]
  elif zone == 4:
    return [1, 2, 3]
  elif zone == 5:
    return [1, 2, 3, 4]

def get_zone_pass_stats(passes):
  passes_normalized = pd.json_normalize(passes['pass'])
  passes['pass_zone'] = passes.apply(lambda row: _get_zone(row['location']), axis=1)
  passes['target_zone'] = passes_normalized.apply(lambda row: _get_zone(row['end_location']), axis=1)
  zone_stats = [(i, {'passes_attempted': 0, 'passes_completed': 0, 'target_zones': defaultdict(lambda: {'passes_attempted': 0, 'passes_completed': 0})}) for i in range(1, 6)]
  zone_stats = dict(zone_stats)
  for pass_attempted in passes.to_dict('records'):
    if pass_attempted['pass_zone'] == -1 or \
        pass_attempted['target_zone'] == -1 or \
        pass_attempted['pass_zone'] == pass_attempted['target_zone'] or \
        pass_attempted['target_zone'] not in valid_targets_per_zone(pass_attempted['pass_zone']):
      continue
    pass_zone = pass_attempted['pass_zone']
    target_zone = pass_attempted['target_zone']
    zone_stats[pass_zone]['passes_attempted'] += 1
    zone_stats[pass_zone]['target_zones'][target_zone]['passes_attempted'] += 1
    if 'outcome' not in pass_attempted['pass'].keys():
      zone_stats[pass_zone]['passes_completed'] += 1
      zone_stats[pass_zone]['target_zones'][target_zone]['passes_completed'] += 1
  total_valid_passes_attempted = sum(zone_stats[i]['passes_attempted'] for i in range(1, 6))
  total_valid_passes_completed = sum(zone_stats[i]['passes_completed'] for i in range(1, 6))
  zone_stats['total_valid_passes_attempted'] = total_valid_passes_attempted
  zone_stats['total_valid_passes_completed'] = total_valid_passes_completed
  return zone_stats

def normalize_zone_name(zone):
  if zone == 1:
    return 'zoneOne'
  elif zone == 2:
    return 'zoneTwo'
  elif zone == 3:
    return 'zoneThree'
  elif zone == 4:
    return 'zoneFour'
  elif zone == 5:
    return 'zoneFive'

def tabulate_probabilities(shots_taken_stats, shots_faced_stats, passes_made_stats, passes_faced_stats):
  zones_probs = [(i, {}) for i in range(1, 6)]
  zones_probs = dict(zones_probs)
  for zone in zones_probs:
    zones_probs[zone]['shootProbability'] = weird_division(shots_taken_stats[zone]['total_shots'], (shots_taken_stats[zone]['total_shots'] + passes_made_stats[zone]['passes_attempted']))
    zones_probs[zone]['blockProbability'] = weird_division(shots_faced_stats[zone]['goal'], shots_faced_stats[zone]['on_target'])
    zones_probs[zone]['shotOnTargetProbability'] = weird_division(shots_taken_stats[zone]['on_target'], shots_taken_stats[zone]['total_shots'])
    
    zones_probs[zone]['interceptProbability'] = weird_division(passes_faced_stats['total_valid_passes_completed'], passes_faced_stats['total_valid_passes_attempted'])
    for target_zone in valid_targets_per_zone(zone):
      zones_probs[zone][f'{normalize_zone_name(zone)}_to_{normalize_zone_name(target_zone)}_attemptPassProbability'] = weird_division(passes_made_stats[zone]['target_zones'][target_zone]['passes_attempted'], 
                                                                                        passes_made_stats[zone]['target_zones'][target_zone]['passes_attempted'] + shots_taken_stats[zone]['total_shots'])
      zones_probs[zone][f'{normalize_zone_name(zone)}_to_{normalize_zone_name(target_zone)}_passAccurateProbability'] = weird_division(passes_made_stats[zone]['target_zones'][target_zone]['passes_completed'], 
                                                                                        passes_made_stats[zone]['target_zones'][target_zone]['passes_attempted'])
  return zones_probs

def convert_to_probabilityStatements(zones_probs):
  statements = []
  team_prefix = FLAGS.team_prefix + '_' if FLAGS.team_prefix is not None else 'teamOne_' if FLAGS.place.lower() == 'home' else 'teamTwo_'
  for zone in range(1, 6):
    zone_probs = zones_probs[zone]
    for prob_name, prob_value in zone_probs.items():
      zone_prefix = ''
      if 'zone' not in prob_name:
        zone_prefix = f'{normalize_zone_name(zone)}_'
      statements.append(f'#define {team_prefix}{zone_prefix}{prob_name} {int(float(round(prob_value, 2)) * 100)};')
  return '\n'.join(statements)

def main(argv):
  print('See flags:')
  print(f'Competition Name: {FLAGS.competition_name}')
  print(f'Team Name: {FLAGS.team_name}')
  print(f'Place: {FLAGS.place}')
  print(f'Team Prefix: {FLAGS.team_prefix}')

  if FLAGS.place.lower() == 'both':
    assert FLAGS.team_prefix is not None, "Team prefix cannot be None if taking both home and away results to account!"

  competition_id, season_id = get_competition_and_season_id(FLAGS.competition_name)
  matches = pd.read_json(f'../data/matches/{competition_id}/{season_id}.json')
  matches = is_team_and_place(matches)
  match_ids = matches['match_id'].to_dict().values()
  print('see match_ids:')
  pprint(match_ids)

  events_in_matches = []
  for match_id in match_ids:
    events_in_match = pd.read_json(f'../data/events/{match_id}.json')
    events_in_matches.append(events_in_match)
  events_in_matches = pd.concat(events_in_matches, axis=0)

  shots_taken_by_team_in_matches = events_in_matches[\
      events_in_matches['type'].apply(check_shot) & \
      events_in_matches['team'].apply(check_is_current_team)\
    ]
  print('see shots taken stats:')
  number_of_shots_taken = len(shots_taken_by_team_in_matches)
  shots_taken_stats = get_zone_shot_stats(shots_taken_by_team_in_matches)
  pprint(shots_taken_stats)
  print('total shots taken: ', number_of_shots_taken)

  passes_made_by_team_in_matches = events_in_matches[\
      events_in_matches['type'].apply(check_pass) & \
      events_in_matches['team'].apply(check_is_current_team)\
    ]
  print('see passes made stats:')
  passes_made_stats = get_zone_pass_stats(passes_made_by_team_in_matches)
  pprint(passes_made_stats)


  shots_faced_by_team_in_matches = events_in_matches[\
      events_in_matches['type'].apply(check_shot) & \
      ~events_in_matches['team'].apply(check_is_current_team)\
    ]
  number_of_shots_faced = len(shots_faced_by_team_in_matches)
  shots_faced_stats = get_zone_shot_stats(shots_faced_by_team_in_matches)
  print('see shots faced stats:')
  pprint(shots_faced_stats)
  print('total shots faced: ', number_of_shots_faced)

  passed_faced_by_team_in_matches = events_in_matches[\
      events_in_matches['type'].apply(check_pass) & \
      ~events_in_matches['team'].apply(check_is_current_team)\
    ]
  passes_faced_stats = get_zone_pass_stats(passed_faced_by_team_in_matches)
  print('see passes faced stats:')
  pprint(passes_faced_stats)

  zones_probs = tabulate_probabilities(shots_taken_stats, shots_faced_stats, passes_made_stats, passes_faced_stats)
  print('see probabilties:')
  pprint(zones_probs)

  probability_statements = convert_to_probabilityStatements(zones_probs)
  print('see probability statements:')
  print(probability_statements)

if __name__ == '__main__':
  app.run(main)