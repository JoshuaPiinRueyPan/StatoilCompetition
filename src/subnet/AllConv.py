import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class AllConv(SubnetBase):
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
			'convW3': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01)),
			'convW4': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)),
			'convW5': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
			'convW6': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)),
			'convW7': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01)),
			'convW8': tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=0.01)),
			'output': tf.Variable(tf.random_normal([256, outSettings.NUMBER_OF_CATEGORIES], stddev=0.01))
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([8], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([32], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([64], stddev=0.01)),
			'convb5': tf.Variable(tf.random_normal([128], stddev=0.01)),
			'convb6': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'convb7': tf.Variable(tf.random_normal([512], stddev=0.01)),
			'convb8': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'output'  : tf.Variable(tf.random_normal([outSettings.NUMBER_OF_CATEGORIES], stddev=0.01)),
		}
		return weights, biases

	def buildNetBody(self, weights, biases):
		conv1 = ConvLayer(self.inputImage, weights['convW1'], biases['convb1'], name='conv1')
		norm1 = AlexNorm(conv1, lsize=4, name='norm1')

		conv2 = ConvLayer(norm1, weights['convW2'], biases['convb2'], name='conv2')
		norm2 = AlexNorm(conv2, lsize=4, name='norm3')

		conv3 = ConvLayer(norm2, weights['convW3'], biases['convb3'], name='conv3')
		pool3 = MaxPoolLayer(conv3, kernelSize=2, name='pool3')
		norm3 = AlexNorm(pool3, lsize=4, name='norm3')

		conv4 = ConvLayer(norm3, weights['convW4'], biases['convb4'], name='conv4')
		pool4 = MaxPoolLayer(conv4, kernelSize=2, name='pool4')
		norm4 = AlexNorm(pool4, lsize=4, name='norm4')

		conv5 = ConvLayer(norm4, weights['convW5'], biases['convb5'], name='conv5')
		pool5 = MaxPoolLayer(conv5, kernelSize=2, name='pool5')
		norm5 = AlexNorm(pool5, lsize=4, name='norm5')

		conv6 = ConvLayer(norm5, weights['convW6'], biases['convb6'], name='conv6')
		pool6 = MaxPoolLayer(conv6, kernelSize=2, name='pool6')
		norm6 = AlexNorm(pool6, lsize=4, name='norm6')

		conv7 = ConvLayer(norm6, weights['convW7'], biases['convb7'], name='conv7')
		norm7 = AlexNorm(conv7, lsize=4, name='norm7')

		conv8 = ConvLayer(norm7, weights['convW8'], biases['convb8'], name='conv8')

		avgpool = tf.nn.avg_pool(conv8, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID', name='avg_pool')
		avgpool = tf.reshape(avgpool, shape=[-1, 256])

		output = tf.add(tf.matmul(avgpool, weights['output']), biases['output'])
		return output

