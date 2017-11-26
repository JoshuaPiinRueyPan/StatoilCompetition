import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class AlexnetTiny(SubnetBase):
	def __init__(self, isTraining_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		weights, biases = self.buildNetVariables()
		return self.buildNetBody(weights, biases)

	def buildNetVariables(self):
		weights = {
			'convW1': tf.Variable(tf.random_normal([3, 3, 2, 8], stddev=0.01)),
			'convW2': tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.01)),
			'convW3': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
			'convW4': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
			'fcW1'  : tf.Variable(tf.random_normal([5*5*16+1, 128], stddev=0.01)),
			'fcW2'  : tf.Variable(tf.random_normal([128, 128], stddev=0.01)),
			'output': tf.Variable(tf.random_normal([128, outSettings.NUMBER_OF_CATEGORIES], stddev=0.01)),
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([8], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'fcb1'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
			'fcb2'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
			'output'  : tf.Variable(tf.random_normal([outSettings.NUMBER_OF_CATEGORIES], stddev=0.01)),
		}
		return weights, biases

	def buildNetBody(self, weights, biases):
		conv1 = ConvLayer(self.inputImage, weights['convW1'], biases['convb1'], name='conv1')
		pool1 = MaxPoolLayer(conv1, kernelSize=2, name='pool1')
		norm1 = AlexNorm(pool1, lsize=4, name='norm1')

		conv2 = ConvLayer(norm1, weights['convW2'], biases['convb2'], name='conv2')
		pool2 = MaxPoolLayer(conv2, kernelSize=2, name='pool2')
		norm2 = AlexNorm(pool2, lsize=4, name='norm3')

		conv3 = ConvLayer(norm2, weights['convW3'], biases['convb3'], name='conv3')
		pool3 = MaxPoolLayer(conv3, kernelSize=2, name='pool3')
		norm3 = AlexNorm(pool3, lsize=4, name='norm3')

		conv4 = ConvLayer(norm3, weights['convW4'], biases['convb4'], padding='VALID', name='conv4')
		pool4 = MaxPoolLayer(conv4, kernelSize=2, name='pool4')
		norm4 = AlexNorm(pool4, lsize=4, name='norm4')

		pool4reshape = tf.reshape(norm4, [-1, weights['fcW1'].get_shape().as_list()[0]-1])
		concat = tf.concat(axis=1, values=[pool4reshape, self.inputAngle])

		fc1 = tf.add(tf.matmul(concat, weights['fcW1']), biases['fcb1'])
		fc1 = tf.nn.relu(fc1)

		dropout = tf.cond(self.isTraining, lambda: 0.5, lambda: 1.)
		fc1 = tf.nn.dropout(fc1, dropout)

		fc2 = tf.reshape(fc1, [-1, weights['fcW2'].get_shape().as_list()[0]])
		fc2 = tf.add(tf.matmul(fc2, weights['fcW2']), biases['fcb2'])
		fc2 = tf.nn.relu(fc2)
		fc2 = tf.nn.dropout(fc2, dropout)

		output = tf.add(tf.matmul(fc2, weights['output']), biases['output'])
		return output

