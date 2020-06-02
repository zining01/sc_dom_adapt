"""Test implementation of CSLS and ISF"""
import numpy as np
from similarity_metrics import csls

def test_csls_1():
	xs = np.array([[1., 1.,], [2., 5.], [-10., -15.]])
	xt = np.array([[1., 1.], [1., 0.5], [3., 6.]])
	k = 2

	model = csls(k)
	print("Source matrix: \n", xs)
	print("Target matrix: \n", xt)

	mtx = model.fit(xs, xt)
	print("Result:\n", mtx)

if __name__=="__main__":
	test_csls_1()