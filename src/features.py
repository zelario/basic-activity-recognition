import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Statistical tests
from scipy import stats


def per_activity_means(trans_data, values):
	activities = np.arange(1, 17)
	means = []
	for act in activities:
		mask = trans_data[:, 11] == act
		vals = values[mask]
		if vals.size == 0:
			means.append(np.nan)
		else:
			means.append(np.mean(vals))
	return activities, np.array(means)


def ks_test(data, values, method='shapiro'):
	results = {}
	for act in range(1, 17):
		mask = data[:, 11] == act
		vals = values[mask]
		if vals.size < 3:
			results[act] = (np.nan, np.nan)
			continue
		if method == 'shapiro':
			stat, p = stats.shapiro(vals)
		elif method == 'kstest':
			# compare empirical distribution to normal with sample mean/std
			mu, sigma = np.mean(vals), np.std(vals, ddof=1)
			if sigma == 0:
				stat, p = (np.nan, np.nan)
			else:
				stat, p = stats.kstest(vals, 'norm', args=(mu, sigma))
		else:
			raise ValueError('method must be shapiro or kstest')
		results[act] = (stat, p)
	return results
