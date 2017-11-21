import tensorflow as tf
import RadarImage

NUMBER_OF_CATEGORIES = 2

def convlayer(name, x, W, b, strides=1, padding='SAME'):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x, name = name)

def maxpool(name, x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


class IceNet:
	def __init__(self):
		self.inputImage = tf.placeholder(tf.float32, [None, RadarImage.DATA_WIDTH, RadarImage.DATA_HEIGHT, RadarImage.DATA_CHANNELS])
		self.inputAngle = tf.placeholder(tf.float32, [None, 1])
		self.groundTruth = tf.placeholder(tf.float32, [None, NUMBER_OF_CATEGORIES])
		self.dropout = tf.placeholder(tf.float32)

	def Build(self):

		weights, biases = self.createVariableTensors()
		netOutput = self.createNetBody(weights, biases)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=netOutput, labels=self.groundTruth))

		accuracy = self.calculateAccuracy(netOutput, self.groundTruth)

		return cost, accuracy

	def createVariableTensors(self):
		weights = {
			'convW1': tf.Variable(tf.random_normal([3, 3, 2, 8], stddev=0.01)),
			'convW2': tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.01)),
			'convW3': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
			'convW4': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
			'fcW1'  : tf.Variable(tf.random_normal([5*5*16+1, 128], stddev=0.01)),
			'fcW2'  : tf.Variable(tf.random_normal([128, 128], stddev=0.01)),
			'output': tf.Variable(tf.random_normal([128, NUMBER_OF_CATEGORIES], stddev=0.01)),
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([8], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'fcb1'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
			'fcb2'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
			'output'  : tf.Variable(tf.random_normal([NUMBER_OF_CATEGORIES], stddev=0.01)),
		}
		return weights, biases

	def createNetBody(self, weights, biases):
		inputx = tf.reshape(self.inputImage, shape=[-1, 75, 75, 2])

		conv1 = convlayer('conv1', inputx, weights['convW1'], biases['convb1'])
		pool1 = maxpool('pool1', conv1, k=2)
		norm1 = norm('norm1', pool1, lsize=4)

		conv2 = convlayer('conv2', norm1, weights['convW2'], biases['convb2'])
		pool2 = maxpool('pool2', conv2, k=2)
		norm2 = norm('norm3', pool2, lsize=4)

		conv3 = convlayer('conv3', norm2, weights['convW3'], biases['convb3'])
		pool3 = maxpool('pool3', conv3, k=2)
		norm3 = norm('norm3', pool3, lsize=4)

		conv4 = convlayer('conv4', norm3, weights['convW4'], biases['convb4'], padding='VALID')
		pool4 = maxpool('pool4', conv4, k=2)
		norm4 = norm('norm4', pool4, lsize=4)

		pool4reshape = tf.reshape(norm4, [-1, weights['fcW1'].get_shape().as_list()[0]-1])
		concat = tf.concat(axis=1, values=[pool4reshape, self.inputAngle])

		fc1 = tf.add(tf.matmul(concat, weights['fcW1']), biases['fcb1'])
		fc1 = tf.nn.relu(fc1)
		fc1 = tf.nn.dropout(fc1, self.dropout)

		fc2 = tf.reshape(fc1, [-1, weights['fcW2'].get_shape().as_list()[0]])
		fc2 = tf.add(tf.matmul(fc2, weights['fcW2']), biases['fcb2'])
		fc2 = tf.nn.relu(fc2)
		fc2 = tf.nn.dropout(fc2, self.dropout)

		output = tf.add(tf.matmul(fc2, weights['output']), biases['output'])
		return output

	def calculateAccuracy(self, netOutput_, groundTruth_):
		correctPredictions = tf.equal(tf.argmax(netOutput_, 1), tf.argmax(groundTruth_, 1))
		correctPredictions = tf.reshape(correctPredictions, shape=[-1])

		accuracy = tf.reduce_mean(tf.cast(correctPredictions, tf.float32))
		return accuracy
