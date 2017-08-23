import tensorflow as tf
import read_storm_data
import math
#import save_network

#parameter
learning_rate = 0.01	 #tf.constant(0.001)
momentum = 0.001
training_epochs = 30	#tf.constant(15)
display_step =  3		#tf.constant(1)
#row_count = 196686;
file_count = 7267;


#network parameter

n_hidden_1 = 20 	 	# number of 1st hidden layer
#n_hidden_2 = 30		# number of 2nd hidden layer
#n_hidden_3 = 10		# number of 3rd hidden layer
#n_hidden_4 = 20		# number of 4rd hidden layer
n_input = 5			# number of input layer
n_class = 1			# number of output layer

factor_d = 0
factor_phi = 1

accept_threshold = 0.05

n_cluster_0 = 278;   n_training_cluster_0 = int(n_cluster_0 * 0.7)
n_cluster_1 = 433;  n_training_cluster_1 = int(n_cluster_1 * 0.7)
n_cluster_2 = 248;  n_training_cluster_2 = int(n_cluster_2 * 0.7)
n_cluster_3 = 138;  n_training_cluster_3 = int(n_cluster_3 * 0.7)
n_cluster_4 = 1002;   n_training_cluster_4 = int(n_cluster_4 * 0.7)
n_cluster_5 = 476;   n_training_cluster_5 = int(n_cluster_5 * 0.7)
n_cluster_6 = 727;  n_training_cluster_6 = int(n_cluster_6 * 0.7)
n_cluster_7 = 587;     n_training_cluster_7 = int(n_cluster_7 * 0.7)
n_cluster_8 = 149;     n_training_cluster_8 = int(n_cluster_8 * 0.7)
n_cluster_9 = 76;     n_training_cluster_9 = int(n_cluster_9 * 0.7)
n_cluster_10 = 393;     n_training_cluster_10 = int(n_cluster_10 * 0.7)
n_cluster_11 = 212;     n_training_cluster_11 = int(n_cluster_11 * 0.7)
n_cluster_12 = 754;     n_training_cluster_12 = int(n_cluster_12 * 0.7)
n_cluster_13 = 77;     n_training_cluster_13 = int(n_cluster_13 * 0.7)
n_cluster_14 = 161;     n_training_cluster_14 = int(n_cluster_14 * 0.7)

n_cluster_test = 93; n_training_cluster_test = int(n_cluster_test * 0.7)


n_training_set = 7000
n_test_set = file_count - n_training_set

#input - output
x = tf.placeholder("float", [None, n_input])	# to load input value 
y = tf.placeholder("float", [None, n_class])	# to load the actual value

# Weights: create dictionary of weights
weights = {
	'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	#'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	#'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	#'h4' : tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
	'out': tf.Variable(tf.random_normal([n_hidden_1, n_class]))
}

# Biases: Create dictionary of biases

biases = {
	'h1' : tf.Variable(tf.random_normal([n_hidden_1])),
	#'h2' : tf.Variable(tf.random_normal([n_hidden_2])),
	#'h3' : tf.Variable(tf.random_normal([n_hidden_3])),
	#'h4' : tf.Variable(tf.random_normal([n_hidden_4])),
	'out': tf.Variable(tf.random_normal([n_class]))
}

# Model

def multi_perceptron(x, weights, biases):
	# use tf.tanh as the activation function
	# return value of each neural in output layer
	
	
	# 1st hidden layer
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])		
	#layer_1 = tf.tanh(layer_1)
	layer_1 = tf.nn.relu(layer_1)
	#layer_1 = tf.nn.relu6(layer_1)	# OK 
	#layer_1 = tf.nn.elu(layer_1)
	#layer_1 = tf.nn.softplus(layer_1)
	#layer_1 = tf.nn.softsign(layer_1)
	#layer_1 = tf.nn.dropout(layer_1)
	#layer_1 = tf.sigmoid(layer_1)
	
	# 2nd hidden layer
	#layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
	#layer_2 = tf.tanh(layer_2)
	#layer_2 = tf.nn.relu(layer_2)
	#layer_2 = tf.nn.relu6(layer_2)	#OK
	#layer_2 = tf.nn.elu(layer_2)
	#layer_2 = tf.nn.softplus(layer_2)
	#layer_2 = tf.nn.softsign(layer_2)
	#layer_2 = tf.nn.dropout(layer_2)
	#layer_2 = tf.sigmoid(layer_2)
	
	#layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
	#layer_3 = tf.tanh(layer_3)
	
	#layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['h4'])
	#layer_4 = tf.tanh(layer_4)
	
	# output layer
	out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

	return out_layer
	

# feed forward
feed_forward = multi_perceptron(x, weights, biases)		# the output value following the model

# calculate loss value: cross-entropy

cost = tf.reduce_mean(tf.abs(feed_forward - y))

# apply gradien descent

#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)	#OK
#optimizer = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)	# OK




# init variables
init = tf.global_variables_initializer()
# sess = tf.Session()

#####
# Main part
with tf.Session() as sess:
	sess.run(init)
	
	for epoch in range(training_epochs):
		total_cost = 0.0
		path_folder = 'G:\\Lab\\storm_data\\cluster_test'
		
		for i in range(n_cluster_test):		# file_count
			vector = read_storm_data.read_data(i+1, path_folder)
						
			len_vector = len(vector)
			
			net_input = [0 for i in range(len_vector)]
			
			for j in range(len_vector):
				net_input[j] = factor_phi*vector[j][0] + factor_d*vector[j][1];
				#net_input[j] = vector[j][0] + vector[j][1];
				
			if (len_vector < n_input + 1):
				continue;
			
			train_x = [[0 for k in range(n_input)] for j in range(len(net_input) - n_input)]
			train_y = [[0 for j in range(1)] for k in range(len(net_input) - n_input)]
			
			for k in range(len(net_input) - n_input):
				for j in range(n_input):
					train_x[k][j] = net_input[j+k];
				train_y[k][0] = net_input[k+n_input];
			
			# run optimizer op
			sess.run(optimizer, feed_dict= {x: train_x, y: train_y})
			
			# run loss op
			avg_cost = sess.run(cost, feed_dict= {x: train_x, y: train_y})
			
			# calculate average cost
			total_cost = total_cost + avg_cost
			
		if epoch % display_step == 0:
			print ("Epoch:" + str(epoch+1)+ " cost: " + str(total_cost))
				
		
	print ("Optimization complete!")
		
	# test model
	accuracy = 0.0
	path_folder = 'G:\\Lab\\storm_data\\cluster_test'
	
	for i in range(n_training_cluster_test, n_cluster_test):		# n_test_set
		vector_test = read_storm_data.read_data(i+1, path_folder)
		#print (str(i) + ' vector: ', vector_test)
		#print ('\n')
		
		len_vector = len(vector_test)
		net_input = [0 for j in range(len_vector)]

		for j in range(len_vector):
			net_input[j] = factor_phi*vector_test[j][0] + factor_d*vector_test[j][1];

	
		if (len_vector < n_input + 1):
			continue;
			
		test_x = [[0 for k in range(n_input)] for j in range(len(net_input) - n_input)]
		test_y = [[0 for j in range(1)] for k in range(len(net_input) - n_input)]
		
		for k in range(len(net_input) - n_input):
			for j in range(n_input):
				test_x[k][j] = net_input[j+k];
			test_y[k][0] = net_input[k+n_input];
		
		
		
		predict = 0.0
		
		#print (test_x)
		
		test_output = sess.run(feed_forward, feed_dict = {x: test_x})
		#print (test_output)
		
		#for i in range(len(test_y)):
		#	print (str(test_output[i]) + ' - ' + str(test_y[i][0])) 
		
		for i in range(len(test_y)):
			if (abs(test_output[i] - test_y[i][0]) < accept_threshold*test_y[i][0]):
				predict = predict + 1
		
		predict = predict/len(test_y)
		#print ("Accuracy: ", predict)
		
		accuracy = accuracy + predict
	"""
	for i in range(10):
		print (str(i) + ' ' + str(test_x[i]) + ' - ' + str(test_y[i][0]))
		"""
	
	print ('Average accuracy: ' , accuracy/(n_cluster_test - n_training_cluster_test))

	# save network 
	
	#save_network.save_network()
	#print ('Saved network!')















 
	

