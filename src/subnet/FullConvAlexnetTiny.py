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
			'convW1': tf.Variable(tf.random_normal([3, 3, 2, 32], stddev=0.01)),
			'convW2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)),
			'convW3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
			'convW4': tf.Variable(tf.random_normal([1, 1, 128, 64], stddev=0.01)),
			'convW5': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01)),
			'convW6': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)),
			'convW1x1' : tf.Variable(tf.random_normal([1, 1, 256, 2], stddev=0.01)),
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([32], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([64], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([128], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([64], stddev=0.01)),
			'convb5': tf.Variable(tf.random_normal([128], stddev=0.01)),
			'convb6': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'convb1x1' : tf.Variable(tf.random_normal([2], stddev=0.01)),
		}
		return weights, biases

	def buildNetBody(self, weights, biases):
		net = ConvLayer(self.inputImage, weights['convW1'], biases['convb1'], name='conv1')
		net = MaxPoolLayer(net, kernelSize=2)
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW2'], biases['convb2'], name='conv2')
		net = MaxPoolLayer(net, kernelSize=2)
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW3'], biases['convb3'], name='conv3')
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW4'], biases['convb4'], name='conv4')
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW5'], biases['convb5'], name='conv5')
		net = MaxPoolLayer(net, kernelSize=2)
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW6'], biases['convb6'], name='conv6')
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW1x1'], biases['convb1x1'], padding='VALID', name='conv1x1')

		print("conv.shape = " + str(net.shape))
		#net = AvgPoolLayer(net, kernelSize=10, name='poolFinal')
		net = MaxPoolLayer(net, kernelSize=10, name='poolFinal')
		output = tf.reshape(net, shape=[-1, 2])

		
		return output

