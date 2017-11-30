import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
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
		net = ConvLayer(self.inputImage, weights['convW1'], biases['convb1'], layerName_='conv1')
		net = MaxPoolLayer(net, kernelSize=2)
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW2'], biases['convb2'], layerName_='conv2')
		net = MaxPoolLayer(net, kernelSize=2)
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW3'], biases['convb3'], layerName_='conv3')
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW4'], biases['convb4'], layerName_='conv4')
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW5'], biases['convb5'], layerName_='conv5')
		net = MaxPoolLayer(net, kernelSize=2)
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW6'], biases['convb6'], layerName_='conv6')
		net = AlexNorm(net, lsize=4)

		net = ConvLayer(net, weights['convW1x1'], biases['convb1x1'], padding='VALID', layerName_='conv1x1')

		print("conv.shape = " + str(net.shape))
		#net = AvgPoolLayer(net, kernelSize=10, layerName_='poolFinal')
		net = MaxPoolLayer(net, kernelSize=10, layerName_='poolFinal')
		output = tf.reshape(net, shape=[-1, 2])

		
		return output

