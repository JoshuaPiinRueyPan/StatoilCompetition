import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.LayerHelper import *
import settings.OutputSettings as outSettings

class AlexnetTinyWithoutAngle(SubnetBase):
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
			'convW1': tf.Variable(tf.random_normal([3, 3, 2, 16], stddev=0.01)),
			'convW2': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01)),
			'convW3': tf.Variable(tf.random_normal([3, 3, 32, 16], stddev=0.01)),
			'convW4': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01)),
			'fcW1'  : tf.Variable(tf.random_normal([9*9*32, 128], stddev=0.01)),
			'fcW2'  : tf.Variable(tf.random_normal([128, 128], stddev=0.01)),
			'output': tf.Variable(tf.random_normal([128, outSettings.NUMBER_OF_CATEGORIES], stddev=0.01)),
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([32], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([32], stddev=0.01)),
			'fcb1'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
			'fcb2'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
			'output'  : tf.Variable(tf.random_normal([outSettings.NUMBER_OF_CATEGORIES], stddev=0.01)),
		}
		return weights, biases

	def buildNetBody(self, weights, biases):
		net = ConvLayer(self.inputImage, weights['convW1'], biases['convb1'], name='conv1')
		#net = MaxPoolLayer(net, kernelSize=2, name='pool1')
		net = AlexNorm(net, lsize=4, name='norm1')

		net = ConvLayer(net, weights['convW2'], biases['convb2'], name='conv2')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')
		net = AlexNorm(net, lsize=4, name='norm3')

		net = ConvLayer(net, weights['convW3'], biases['convb3'], name='conv3')
		net = MaxPoolLayer(net, kernelSize=2, name='pool3')
		net = AlexNorm(net, lsize=4, name='norm3')

		net = ConvLayer(net, weights['convW4'], biases['convb4'], padding='VALID', name='conv4')
		net = MaxPoolLayer(net, kernelSize=2, name='pool4')
		net = AlexNorm(net, lsize=4, name='norm4')

		net = tf.reshape(net, [-1, weights['fcW1'].get_shape().as_list()[0]])

		net = tf.add(tf.matmul(net, weights['fcW1']), biases['fcb1'])
		net = tf.nn.relu(net)

		dropout = tf.cond(self.isTraining, lambda: 0.5, lambda: 1.)
		net = tf.nn.dropout(net, dropout)

		net = tf.reshape(net, [-1, weights['fcW2'].get_shape().as_list()[0]])
		net = tf.add(tf.matmul(net, weights['fcW2']), biases['fcb2'])
		net = tf.nn.relu(net)
		net = tf.nn.dropout(net, dropout)

		output = tf.add(tf.matmul(net, weights['output']), biases['output'])
		return output

