import typing

import matplotlib.pyplot as plt
import numpy as np


def generate_dataset(N=1024, distribution: typing.Literal["centroids", "ring", "spiral"] = 'centroids', seed=42, plot=True):
	"""
	Generate a synthetic dataset with two classes for binary classification.

	Arguments:
	----------
	N : int, optional
		Number of samples in the dataset. The default is 1024.
	distribution : string, optional
		Type of distribution to use.
		One of {'centroids', 'ring', 'spiral'}.
		The default is 'centroids'.
	seed : int, optional
		Random seed for reproducibility. The default is 42.
	plot : bool, optional
		Whether to plot the dataset. The default is True.
	"""
	X = np.zeros((N, 2))
	y = np.zeros(N)
	distribution = distribution.lower()

	if distribution == "centroids":
		X[:N // 2] = np.random.normal(loc=[-1, -1], scale=0.5, size=(N // 2, 2))
		X[N // 2:] = np.random.normal(loc=[1, 1], scale=0.5, size=(N // 2, 2))
		y[:N // 2] = 0
		y[N // 2:] = 1

	elif distribution == "ring":
		X[:N // 2] = np.random.normal(loc=[0, 0], scale=0.5, size=(N // 2, 2))
		y[:N // 2] = 0
		for i in range(N // 2):
			radius = np.random.uniform(low=1.75, high=3)
			angle = np.random.uniform(low=0, high=2 * np.pi)
			X[N // 2 + i, 0] = radius * np.cos(angle)
			X[N // 2 + i, 1] = radius * np.sin(angle)
		y[N // 2:] = 1

	elif distribution == "spiral":
		for i in range(N // 2):
			r = i / (N // 2 - 1)
			t = 3 * i / (N // 2 - 1) * np.pi
			X[i] = np.array([np.cos(t) * r, np.sin(t) * r]) + np.random.normal(loc=[0, 0], scale=0.05)
			y[i] = 0
			r = i / (N // 2 - 1)
			t = 3 * i / (N // 2 - 1) * np.pi + np.pi
			X[i + N // 2] = np.array([np.cos(t) * r, np.sin(t) * r]) + np.random.normal(loc=[0, 0], scale=0.05)
			y[i + N // 2] = 1

	else:
		print("Invalid distribution specified!")
		return None, None

	if plot:
		np.random.seed(seed)
		plt.figure(figsize=(6, 6))
		plt.scatter(
			X[y == 0, 0], X[y == 0, 1],
			label='Class 0',
			marker='o',
			edgecolors='k'
		)
		plt.scatter(
			X[y == 1, 0], X[y == 1, 1],
			label='Class 1',
			marker='s',
			edgecolors='k'
		)
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.legend()
		plt.show()

	return X, y


def plot_decision_boundary(model, X, Y):
	"""
	Plots the decision boundary of a given model along with the data points.

	Arguments:
	----------
	model (object): The model used to predict the class labels.
		It should have a `forward` method that takes input features and returns predicted probabilities.
	X (numpy.ndarray): The input features, a 2D array of shape (n_samples, 2).
	Y (numpy.ndarray): The true class labels, a 1D array of shape (n_samples,).
	"""
	# Create a meshgrid of points to evaluate the model
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
						 np.arange(y_min, y_max, 0.01))

	# Predict the class for each point in the meshgrid
	Z = model.forward(np.c_[xx.ravel(), yy.ravel()].T)
	Z = (Z > 0.5).astype(int)  # Assuming binary classification
	Z = Z.reshape(xx.shape)

	# compute accuracy
	acc = np.mean(np.round(model.forward(X.T)) == Y)

	# Plot the decision boundary and data points
	plt.figure(figsize=(6, 6))
	plt.contourf(xx, yy, Z, alpha=0.5)
	plt.scatter(X[Y == 0, 0], X[Y == 0, 1], label='Class 0', marker='o', edgecolors='k')
	plt.scatter(X[Y == 1, 0], X[Y == 1, 1], label='Class 1', marker='s', edgecolors='k')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.legend()
	plt.title(f"Decision Boundary\nAccuracy: {acc * 100:.2f}%")
	plt.show()


from sklearn.model_selection import train_test_split


def split_dataset(X, Y, test_size=0.2, val_size=0.1, random_state=42):
	"""Splits a dataset into training, validation, and testing sets.

	Arguments:
	----------
	X (numpy.ndarray): The input features.
	y (numpy.ndarray): The target variable.
	test_size (float): The proportion of the dataset to include in the test split.
	val_size (float): The proportion of the dataset to include in the validation split.
	random_state (int): Controls the shuffling applied to the data before applying the split.

	Returns:
		tuple: A tuple containing the training, validation, and testing sets as NumPy arrays:
			(X_train, X_val, X_test, y_train, y_val, y_test)
	"""
	X_train, X_temp, Y_train, Y_temp = train_test_split(
		X, Y, test_size=test_size + val_size, random_state=random_state
	)

	X_val, X_test, Y_val, Y_test = train_test_split(
		X_temp, Y_temp, test_size=test_size / (test_size + val_size), random_state=random_state
	)

	return X_train, Y_train, X_val, Y_val, X_test, Y_test


def plot_activation_function(forward_function, derivative_function, title):
	"""
	Plots an activation function and its derivative.

	Arguments:
	----------
	forward_function (function): The activation function to be plotted.
	derivative_function (function): The derivative of the activation function.
	title (str): The title of the plot.
	"""

	z = np.linspace(-10, 10, 100)
	func_output = forward_function(z)
	func_derivative = derivative_function(func_output)

	plt.figure(figsize=(8, 6))
	plt.plot(z, func_output, label='Activation Function')
	plt.plot(z, func_derivative, label='Derivative of Function')
	plt.xlabel('z')
	plt.ylabel('Function Value')
	plt.title(title)
	plt.legend()
	plt.grid()
	plt.show()


def plot_history(history):
	"""
	Plots the training and validation loss, and optionally the learning rate and accuracy, from a given history dictionary.

	Arguments:
	----------
	history (dict): A dictionary containing the training history. It should have the following keys:
		- 'loss': A list of training loss values.
		- 'val_loss': A list of validation loss values.
		- 'learning_rate' (optional): A list of learning rate values.
		- 'accuracy' (optional): A list of training accuracy values.
		- 'val_accuracy' (optional): A list of validation accuracy values.
	"""
	# Function implementation goes here
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=120)
	ax1.plot(history['loss'], label='training')
	ax1.plot(history['val_loss'], label='validation', linestyle='--')
	ax1.set_yscale("log")
	ax1.set_ylabel('Loss')
	if "learning_rate" in history.keys():
		ax1b = ax1.twinx()
		ax1b.plot(history['learning_rate'], 'g-', linewidth=1)
		ax1b.set_yscale('log')
		ax1b.set_ylabel('Learning Rate', color='g')

	if "accuracy" in history.keys():
		ax2.plot(history['accuracy'], label='training')
		ax2.plot(history['val_accuracy'], label='validation', linestyle='--')
		ax2.set_ylabel('Accuracy')
		ax2.set_xlabel('Epochs')
		ax2.legend()
	plt.show()


class Layer:
	"""
	A base class for neural network layers.

	Attributes:
	----------
	_name : str
		The name of the layer, default is 'BaseLayer'.

	Methods:
	-------
	__init__():
		Initializes the layer.

	forward(input):
		Computes the output of the layer for a given input.
		This method should be implemented by subclasses.

	backward(output_error, learning_rate):
		Computes the gradient of the loss with respect to the input.
		This method should be implemented by subclasses.

	update(learning_rate):
		Updates the parameters of the layer using the given learning rate.

	number_of_parameters:
		Returns the number of parameters in the layer.
	"""

	_name = 'BaseLayer'

	def forward(self, input):
		raise NotImplementedError

	def backward(self, output_error, learning_rate):
		raise NotImplementedError

	def update(self, learning_rate):
		pass

	@property
	def number_of_parameters(self):
		if not hasattr(self, '_number_of_parameters'):
			if hasattr(self, 'W') & hasattr(self, 'b'):
				self._number_of_parameters = self.W.size + self.b.size
			else:
				self._number_of_parameters = 0
		return self._number_of_parameters
