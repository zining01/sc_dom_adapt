import numpy as np

def transfer_labels(match, target_labels):
	"""Given coupling matrix and target labels,
	compute most likely source labels

	Labels need to be integer-encoded

	(Uses soft transfer as a helper function)

	Inputs:
	match (ns * nt)
	target_labels (nt,)

	Returns:
	source_labels (ns,)
	"""
	label_distribution = transfer_labels_soft(match, target_labels)

	# returns argmax of label distribution
	return np.argmax(label_distribution, axis=1)

def transfer_labels_soft(match, target_labels):
	"""Given coupling matrix and target labels,
	comput distribution over labels for each point in source

	Labels need to be integer-encoded

	Inputs:
	match (ns * nt)
	target_labels (nt,)

	Returns:
	label_distribution (ns, n_classes)
	"""
	# get shape of input
	ns, nt = match.shape

	# get number of target labels
	n_classes = np.max(target_labels) + 1

	# create one-hot representation of target
	target_one_hot = np.zeros((nt, n_classes))
	target_one_hot[np.arange(nt),target_labels] = 1.

	# get distribution over labels
	label_distribution = match @ target_one_hot #/ match.sum(axis=1, keepdims=True)

	return label_distribution

def create_cluster_heatmap(match, source_labels, target_labels):
	"""Given coupling matrix and source/target labels
	compute map between source and target clusters

	Labels for source and target need to be integer-encoded

	Inputs:
	match (ns * nt)
	source_labels (ns, )
	target_labels (nt, )

	Returns:
	Map from source to target classes (n_source_cl, n_target_cl)
	"""
	# get matrix dimensions
	ns, nt = match.shape

	# number of classes
	n_classes_s = np.max(source_labels) + 1
	n_classes_t = np.max(target_labels) + 1

	# get one-hot encoding of source labels
	source_one_hot = np.zeros((ns, n_classes_s))
	source_one_hot[np.arange(ns), source_labels] = 1.

	# normalize one-hot so that every column sums to 1
	source_distribution = source_one_hot #/ source_one_hot.sum(axis=0, keepdims=True)

	# get target label distribution
	label_distribution = transfer_labels_soft(match, target_labels)
	#label_distribution = label_distribution / match.sum(axis=1, keepdims=True)

	# get joint distribution
	joint = source_distribution.T @ label_distribution

	return joint
