import tensorflow as tf
import numpy as np



n_hidden_1 = 256 		# number of 1st hidden layer
n_hidden_2 = 256		# number of 2nd hidden layer
n_input = 784			# number of input layer
n_class = 1			# number of output layer

weights = {
	'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_class]))
}

biases = {
	'h1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'h2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_class]))
}
	

save_weights = {
	'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_class]))
}
	
np.save('weights.npy',weights)
np.save('biases.npy', biases)
