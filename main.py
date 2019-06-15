import numpy as np
import pandas as pd
from scipy import stats

import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# Helper function
def update_win_dict(win_dict, update):
	for key in update:
		if key not in win_dict:
			win_dict[key] = update[key]
		else:
			win_dict[key].extend(update[key])
	return win_dict


def make_teams(n = 128, param1_dist = None, param2_dist = None):
	""" Sample means/variances """

	# Create prior dists
	if param1_dist is None:
		param1_dist = stats.gamma(a = 9, scale = 0.5)
	if param2_dist is None:
		param2_dist = stats.beta(a = 2, b = 5)

	# Sample teams
	p1s = param1_dist.rvs(n)
	p2s = 5*param2_dist.rvs(n)

	return p1s, p2s


def prelim_round_bracket(teams, num_wins, sample_dist, high_low):
	""" For a group of teams with the same number of wins, pairs rounds """

	# Pair rounds
	num_teams = len(teams)
	if num_teams % 2 != 0:
		raise ValueError('Number of teams must be divisible by two within a bracket; pullups not implemented')
	num_rounds = int(num_teams/2)
	if high_low:
		teams = sorted(teams, key = lambda x: x[3])
	else:
		np.random.shuffle(teams)
	team1s = teams[0:num_rounds]
	team2s = teams[num_rounds:]
	team2s.reverse()

	# Simulate rounds
	winners = []
	losers = []
	for i in range(num_rounds):
		team1_val = sample_dist.rvs(team1s[i][1], team1s[i][2], size = 1)[0]
		team1s[i][3] += team1_val
		team2_val = sample_dist.rvs(team2s[i][1], team2s[i][2], size = 1)[0]
		team2s[i][3] += team2_val
		if team1_val > team2_val:
			winners.append(team1s[i])
			losers.append(team2s[i])
		else:
			winners.append(team2s[i])
			losers.append(team1s[i])

	return {num_wins+1:winners, num_wins:losers}

def prelim_round(wins_tracker, sample_dist, high_low):
	"""
	wins_tracker: dict mapping num_wins to list of teams with that num wins
	**kwargs: **kwargs to prelim_round_bracket
	"""
	new_wins_tracker = {}
	for key in wins_tracker:
		update = prelim_round_bracket(teams = wins_tracker[key],
									  num_wins = key,
									  sample_dist = sample_dist,
									  high_low = high_low)
		new_wins_tracker = update_win_dict(new_wins_tracker, update)

	return new_wins_tracker


def run_tournament(p1s = None, p2s = None, sample_dist = None, 
				   num_prelims = 7, break_rounds = 4,
				   high_low = True, **kwargs):

	# Get parameters and sample_dist
	if p1s is None or p2s is None:
		p1s, p2s = make_teams(**kwargs)
	else:
		if len(p1s) != len(p2s):
			raise IndexError('Means and variances are not equal length')
	if sample_dist is None:
		sample_dist = stats.norm

	# Data for teams
	num_teams = len(p1s)
	codes = np.arange(0, num_teams, 1)
	# Teams have: code, param1, param2, performance
	teams = [[c, p1, np.sqrt(p2), 0] for c, p1, p2 in zip(codes, p1s, p2s)]
	wins_tracker = {0:teams}

	# Run prelim rounds --------------------------------------
	for i in range(num_prelims):
		wins_tracker = prelim_round(wins_tracker, sample_dist, high_low = high_low)


	# Construct break ----------------------------------------
	num_break = 2**break_rounds
	breaking_teams = []
	max_wins = num_prelims
	while True:

		# Sort by performance
		wins_tracker[max_wins] = sorted(wins_tracker[max_wins], 
										key = lambda x: -1*x[3])

		# Add the number left
		num_left = num_break - len(breaking_teams)
		if len(wins_tracker[max_wins]) < num_left:
			breaking_teams.extend(wins_tracker[max_wins])
			max_wins -= 1
		else:
			breaking_teams.extend(
				wins_tracker[max_wins][0:num_left]
			)
			break

	# Start to collect results
	result_df = pd.DataFrame(index = codes)
	result_df['p1'] = p1s
	result_df['p2'] = p2s
	wins = pd.Series()
	performances = pd.Series()
	for key in wins_tracker:
		codes = [x[0] for x in wins_tracker[key]]
		perfs = [x[3] for x in wins_tracker[key]]
		wins = wins.append(
			pd.Series([key]*len(codes), index = codes)
		)
		performances = performances.append(
			pd.Series(perfs, index = codes)
		)

	result_df['wins'] = wins
	result_df['performance'] = performances

	# Add seeding
	result_df = result_df.sort_values(
		by = ['wins', 'performance'], ascending = False
	)
	result_df['seed'] = np.arange(1, num_teams + 1, 1)
	result_df['outround'] = 0

	# Simulate break rounds ---------------------------
	moving_on = breaking_teams
	for i in range(break_rounds + 1):

		# Codes
		codes = [x[0] for x in moving_on]
		result_df.loc[codes, 'outround'] = i + 1
		if i == break_rounds:
			assert len(codes) == 1, 'Tournament over but not one winner'
			break
		
		# Simulate break round
		num_rounds = int(len(codes)/2)
		team1s = moving_on[0:num_rounds]
		team2s = moving_on[num_rounds:]
		team2s.reverse()
		winners = []
		for j in range(num_rounds):
			team1_val = sample_dist.rvs(team1s[j][1], team1s[j][2], size = 1)[0]
			team2_val = sample_dist.rvs(team2s[j][1], team2s[j][2], size = 1)[0]
			if team1_val > team2_val:
				winners.append(team1s[j])
			else:
				winners.append(team2s[j])

		# Update teams which are moving on
		moving_on = winners

	return result_df

def plot_result_df(result_df):

	#outround_ranking = sorted(result_df['outround'].unique())
	#outround_ranking = [str(x) for x in outround_ranking]
	#result_df['outround'] = result_df['outround'].astype(str)
	#print(result_df)
	result_df['variance'] = np.sqrt(result_df['p2'])


	sns.scatterplot(x = 'p1', y = 'outround',
					size = 'variance',
					hue = 'variance',
					data = result_df)
	plt.show()

def simulate_multiple_tournaments(n = 16, num_trials = 50, **kwargs):

	p1s, p2s = make_teams(n = n)

	all_results = run_tournament(p1s = p1s, p2s = p2s, 
								 **kwargs)
	all_results['trial'] = 0
	all_results['code'] = all_results.index
	all_results.reset_index(inplace = True, drop = True)

	for s in range(num_trials - 1):

		result_df = run_tournament(p1s = p1s, p2s = p2s, 
								   **kwargs)
		result_df['code'] = result_df.index
		result_df['trial'] = s + 1
		result_df.reset_index(inplace = True, drop = True)
		all_results = pd.concat([all_results, result_df], sort = False)

	return all_results


def season_results(n = 64, num_trials = 50, **kwargs):
	all_results = simulate_multiple_tournaments(n = n, 
												num_trials = num_trials,
												**kwargs)
	teams = all_results.loc[all_results['trial'] == 0,
							['p1', 'p2', 'code']].set_index('code')

	# Outround counts
	outround_counts = all_results.groupby(['code'])['outround'].value_counts()
	outround_counts = outround_counts.unstack().fillna(0)
	outround_counts.columns = ['outround_' + str(x) for x in outround_counts.columns]

	team_results = pd.concat([teams, outround_counts], axis = 1)
	team_results = team_results.sort_values('p1', ascending = False)
	team_results['rank'] = np.arange(1, n+1, 1)
	return team_results

def main(seasons = 1000):

	all_results = season_results(num_trials = 1, break_rounds = 3, num_prelims = 5)
	all_results['season'] = 0
	for i in tqdm(range(seasons - 1)):
		new_results = season_results(num_trials = 1, break_rounds = 3, num_prelims = 5)
		new_results['season'] = i + 1
		all_results = pd.concat([new_results, all_results])

	break_chance = 1 - all_results.groupby(['rank'])['outround_0'].mean()
	fig, ax = plt.subplots()
	sns.scatterplot(break_chance.index, break_chance.values, ax = ax)
	sns.lineplot(break_chance.index, break_chance.values, alpha = 0.5, ax = ax)
	ax.set(title = 'Rank at Tournament vs. Chance of Breaking',
			xlabel = 'True Rank', ylabel = 'Chance of Breaking')
	plt.show()
	return all_results

if __name__ == '__main__':

	main()