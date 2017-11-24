import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.LayerHelper import *
import settings.OutputSettings as outSettings

class FullConvAlexnetTiny(SubnetBase):
	'''
	    Summary:
		This net fail to give good results.  The final accuracy is ONLY 0.5.
		Looks like Fc after Conv Net is important.  Otherwise the net will
		have too few parameters to learn the task

	    Note: In the last layer, both the AvgPool and MaxPool are worse here.
	'''
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
			'convW1x1' : tf.Variable(tf.random_normal([1, 1, 16, 2], stddev=0.01)),
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([8], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([16], stddev=0.01)),
			'convb1x1' : tf.Variable(tf.random_normal([2], stddev=0.01)),
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

		conv4 = ConvLayer(norm3, weights['convW4'], biases['convb4'], name='conv4')
		pool4 = MaxPoolLayer(conv4, kernelSize=2, name='pool4')
		norm4 = AlexNorm(pool4, lsize=4, name='norm4')

		conv1x1 = ConvLayer(norm4, weights['convW1x1'], biases['convb1x1'], padding='VALID', name='conv1x1')
		avgPoolFinal = AvgPoolLayer(conv1x1, kernelSize=5, name='poolFinal')
		output = tf.reshape(avgPoolFinal, shape=[-1, 2])

		#maxPoolFinal = MaxPoolLayer('poolFinal', conv1x1, kernelSize=5)
		#output = tf.reshape(maxPoolFinal, shape=[-1, 2])
		
		return output

