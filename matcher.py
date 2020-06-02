import numpy as np
from numpy.linalg import svd
from sklearn.neighbors import NearestNeighbors

class matcher:
	"""Method for aligning multi-omic data sets.
	Based on VECMAP for cross-lingual alignment of word embeddings

	Attributes:
		k: number of neighbors
		verbose: verbosity
	"""

	def __init__(self, k:int, verbose:bool=True):
		self.k = k
		self.verbose = verbose

	def _compute_dictionary(self, X, Y, W1, W2):
		"""Compute dictionary based on kNN in joint embedding space

		X (n1 x d1): domain 1 embedding
		Y (n2 x d2): domain 2 embedding
		W1 (d1 x d): linear transformation for domain 1
		W2 (d2 x d): linear transformation for domain 2
		"""
		X_joint = X @ W1
		Y_joint = Y @ W2

		if self.verbose:
			print("Initializing ball-tree...")

		# create kNN object
		X_nn = NearestNeighbors(n_neighbors=self.k, algorithm="ball_tree", metric="euclidean")
		X_nn.fit(X_joint)

		Y_nn = NearestNeighbors(n_neighbors=self.k, algorithm="ball_tree", metric="euclidean")
		Y_nn.fit(Y_joint)

		if self.verbose:
			print("Computing kNN graph...")

		# query object to get kNN graph. shape (n2 x n1). CSR matrices.
		y_to_x_nn = X_nn.kneighbors_graph(Y_joint, mode="connectivity")
		x_to_y_nn = Y_nn.kneighbors_graph(X_joint, mode="connectivity")

		# mutually nearest neighbors by element-wise multiplication
		if self.verbose:
			print("Computing mutually nearest neighbors...")
		mnn = x_to_y_nn.multiply(y_to_x_nn.T)

		return mnn

	def _update_transformation(self, X, Y, D):
		"""Given dictionary D, update transformation between X and Y

		X (n1 x d1): domain 1 embedding
		Y (n2 x d2): domain 2 embedding
		D (n1 x n2): mapping (sparse matrix, huge
		"""
		sim_mtx = X.T @ D @ Y
		u, s, v = svd(sim_mtx)

		return u, v.T

	def optimize(self, X, Y, W1, W2, n_iter=1):
		"""Alternate between MNN and SVD"""

		for iteration in range(n_iter):
			mnn = self._compute_dictionary(X, Y, W1, W2)
			W1, W2 = self._update_transformation(X, Y, mnn)

		return mnn, W1, W2

	def compute_translation(self, X, Y, W1, W2, k:int=5):
		"""get k-nearest neighbors across domains

		X (n1 x d1): domain 1 embedding
		Y (n2 x d2): domain 2 embedding
		W1 (d1 x d): linear transformation for domain 1
		W2 (d2 x d): linear transformation for domain 2
		"""
		X_joint = X @ W1
		Y_joint = Y @ W2

		if self.verbose:
			print("Initializing ball-tree...")

		# create kNN object
		X_nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean")
		X_nn.fit(X_joint)

		Y_nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean")
		Y_nn.fit(Y_joint)

		if self.verbose:
			print("Computing kNN graph...")

		# query object to get kNN graph. shape (n2 x n1). CSR matrices.
		y_to_x_nn = X_nn.kneighbors_graph(Y_joint, mode="connectivity")
		x_to_y_nn = Y_nn.kneighbors_graph(X_joint, mode="connectivity")

		return x_to_y_nn, y_to_x_nn


