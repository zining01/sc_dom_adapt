"""Implementation of CSLS and ISF distance matrix adjustment formulas
"""
import numpy as np
from scipy.spatial.distance import cdist

class csls:
	"""Cross-domain similarity with local scaling

	Uses pairwise similarity matrices that become dense (not suitable for large sample sizes)

	Attributes:
		k: number of nearest neighbors for discounting
	"""

	def __init__(self, k:int):
		"""Initialize a new model

		Inputs:
			k (int): number of nearest neighbors

		Returns:
			adj_cosine_sim: adjusted similarity matrix between source and target
		"""
		self.k = k

	def fit(self, xs, xt):
		"""Fit model for source and target coordinates and return similarity matrix
		"""
		# compute cosine similarity (need to subtract 1 because want similarity, not distance)
		cosine_sim = 1. - cdist(xs, xt, metric="cosine")

		# compute average nearest neighbors similarity
		# first need to compute mask to get the top points in each row
		rs_mask = (np.argsort(np.argsort(-cosine_sim, axis=1), axis=1) < self.k).astype(int)
		rt_mask = (np.argsort(np.argsort(-cosine_sim, axis=0), axis=0) < self.k).astype(int)

		# get average nn similarities
		rs = np.sum(cosine_sim * rs_mask, axis=1, keepdims=True) / self.k
		rt = np.sum(cosine_sim * rt_mask, axis=0, keepdims=True) / self.k

		# adjust similarity matrix
		adj_cosine_sim = 2 * cosine_sim - rs - rt

		return adj_cosine_sim

class isf:
	"""Inverted Softmax Function for similarity

	Attributes:
		beta: temperature parameter
	"""

	def __init__(self, beta:float=1.):
		assert isinstance(beta, float), "beta must be float"
		assert beta > 0., "beta must be greater than zero"

		self.beta = beta

	def fit(self, xs, xt):
		"""Returns adjusted similarity matrix"""
		pass


		

		