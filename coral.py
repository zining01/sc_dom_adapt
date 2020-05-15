import numpy as np
from scipy.sparse import csr_matrix

class coral:
	"""An implementation of CORAL for unsupervised domain adaptation
	https://arxiv.org/pdf/1511.05547.pdf
	
	Attributes:
		alpha (float): regularization hyperparameter
		weights (np.ndarray or None): weight of each example
		verbose (bool): verbosity

		whitening (np.ndarray): source whitening matrix
		recoloring (np.ndarray): target recoloring matrix
	"""

	def __init__(self, alpha:float=1., verbose:bool=False):
		self.alpha = alpha
		self.verbose = verbose

	def __repr__(self):
		s1 = "CORAL model: alpha=%.6f" % (self.alpha)
		return s1

	def __str__(self):
		return self.__repr__()

	def _compute_covariance_sp(self, X):
		"""Compute covariance of data in X, where X is a CSR matrix
		"""
		# get shape of matrix
		n, d = X.shape

		# sum of X - mu
		C = ((X.T @ X - (sum(X).T @ sum(X)/n))/(n-1)).todense()
		return C

	def _compute_weighted_covariance_sp(self, X, w):
		"""Compute weighted covariance matrix
		Assumes that the sum of weights (given by w) is equal to number of samples n

		Inputs:
			X (CSR matrix)
			w (np.ndarray)
		"""
		# get shape of matrix
		n, d = X.shape

		weighted_X = X.multiply(w.reshape(-1, 1)).tocsr()

		# computes covariance without densifying the matrix
		C = ((weighted_X.T @ X - (sum(weighted_X).T @ sum(X)/n))/(n-1)).todense()
		return C

	def _inv_square_root(self, S):
		"""Inverse square root of  positive definite matrix S

		S should be a dense NumPy array.
		"""
		# compute  eigendecomposition
		w, v = np.linalg.eigh(S)
		return v @ np.diag(w**(-0.5)) @ v.T

	def _square_root(self, S):
		"""Compute square root of positive definite matrix S"""
		# compute  eigendecomposition
		w, v = np.linalg.eigh(S)
		return v @ np.diag(w**(0.5)) @ v.T

	def fit(self, X, Y, w1=None, w2=None):
		"""Fit model for source X and target Y

		Inputs:
			X (np.ndarray): shape (n1 x d)
			Y (np.ndarray): shape (n2 x d)
			w1 (np.ndarray): shape (n1,)
			w2 (np.ndarray): shape (n2,)
		"""
		# get matrix shapes
		n1, d1 = X.shape
		n2, d2 = Y.shape

		# make sure matrix dims equal
		assert d1 == d2, "X and Y should have equal number of columns"


		if w1 is None and w2 is None:
			Cs = self._compute_covariance_sp(X) + np.eye(d1) * self.alpha
			Ct = self._compute_covariance_sp(Y) + np.eye(d2) * self.alpha

		else:
			# make sure weight vector has correct length
			assert n1 == len(w1), "Length of weight vector w1 not equal to n1"
			assert n2 == len(w2), "Length of weight vector w2 not equal to n2"
			
			# make sure weights have sum n1, n2
			w1 = w1 * (n1/np.sum(w1))
			w2 = w2 * (n2/np.sum(w2))

			Cs = self._compute_weighted_covariance_sp(X, w1) + np.eye(d1) * self.alpha
			Ct = self._compute_weighted_covariance_sp(Y, w2) + np.eye(d2) * self.alpha

		# store sample weights for source and target
		self.w1 = w1 #source weights
		self.w2 = w2 #target weights

		# store whitening and recoloring matrix
		self.whitening = self._inv_square_root(Cs)
		self.recoloring = self._square_root(Ct)

	def fit_transform(self, X, Y, w1=None, w2=None):
		"""Fit model for source X and target Y
		Return transformation of source to target domain

		Inputs:
			X (np.ndarray): shape (n1 x d)
			Y (np.ndarray): shape (n2 x d)
			w1 (np.ndarray or None): shape (n1,)
			w2 (np.ndarray or None): shape (n2,)
		Returns:
			X_transformed (np.ndarray): shape (n1 x d)
		"""
		self.fit(X, Y, w1, w2)

		# note that this will densify matrix
		return X @ self.whitening @ self.recoloring

	def transform(self, X):
		"""Using precomputed covariance matrices, transform new matrix X to target domain

		Inputs:
			X (np.ndarray): shape (n1 x d)

		Returns:
			X_transformed (np.ndarray): shape (n1 x d)
		"""
		# get matrix dimensions
		n, d = X.shape

		# check dimensions
		assert d == self.whitening.shape[0], "X has %d columns but should have %d columns" % (d, self.whitening.shape[0])

		# return transformed matrix
		return X @ self.whitening @ self.recoloring
