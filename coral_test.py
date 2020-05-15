import numpy as np
from scipy.sparse import csr_matrix

from coral import coral

def test_1():
	""""""
	print("Testing covariance computation...")
	X = csr_matrix([[1, 3, 0, 4], [0, 1, 0, 2], [10, 0,1, -3]])
	model = coral(alpha=1., weights=None, verbose=True)
	cov = model._compute_covariance_sp(X)

	print("Covariance computed by CORAL: ")
	print(cov)

	print("Covariance computed by NumPy")
	print(np.cov(X.T.todense()))

def test_2():
	print("Testing square root computation...")

	X = csr_matrix([[1, 3, 0, 4], [0, 1, 0, 2], [10, 0,1, -3]])
	model = coral(alpha=1., weights=None, verbose=True)
	cov = model._compute_covariance_sp(X)

	S = cov + np.eye(4)

	print("Square root: ")
	S_sqrt =  model._square_root(S)
	print(S_sqrt)

	print("S: ")
	print(S)

	print("Reconstr:")
	print(S_sqrt @ S_sqrt)

def test_3():
	print("Testing inverse square root computation...")

	X = csr_matrix([[1, 3, 0, 4], [0, 1, 0, 2], [10, 0,1, -3]])
	model = coral(alpha=1., verbose=True)
	cov = model._compute_covariance_sp(X)

	S = cov + np.eye(4)

	print("Inverse square root: ")
	S_sqrt =  model._inv_square_root(S)
	print(S_sqrt)

	print("S: ")
	print(S)

	print("S * S^-0.5 * S^-0.5:")
	print(S @ S_sqrt @ S_sqrt)

def test_4():
	""""""
	print("Testing weighted covariance computation...")
	X = csr_matrix([[1, 3, 0, 4], [0, 1, 0, 2], [10, 0,1, -3]])
	w1 = np.array([2,1,1])

	model = coral(alpha=1., verbose=True)
	cov = model._compute_weighted_covariance_sp(X, w1)

	print("Weighted covariance computed by CORAL: ")
	print(cov)

	print("Unweighted covariance computed by NumPy")
	print(np.cov(X.T.todense()))

if __name__=="__main__":
	test_4()