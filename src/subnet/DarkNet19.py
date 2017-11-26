import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class DarkNet19(SubnetBase):
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
			'convW7': tf.Variable(tf.random_normal([1, 1, 256, 128], stddev=0.01)),
			'convW8': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01)),
			'convW9': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01)),
			'convW10': tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=0.01)),
			'convW11': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01)),
			'convW12': tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=0.01)),
			'convW13': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01)),
			'convW14': tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=0.01)),
			'convW15': tf.Variable(tf.random_normal([1, 1, 1024, 512], stddev=0.01)),
			'convW16': tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=0.01)),
			'convW17': tf.Variable(tf.random_normal([1, 1, 1024, 512], stddev=0.01)),
			'convW18': tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=0.01)),
			'convW19': tf.Variable(tf.random_normal([1, 1, 1024, 2], stddev=0.01)),
		}

		biases = {
			'convb1': tf.Variable(tf.random_normal([32], stddev=0.01)),
			'convb2': tf.Variable(tf.random_normal([64], stddev=0.01)),
			'convb3': tf.Variable(tf.random_normal([128], stddev=0.01)),
			'convb4': tf.Variable(tf.random_normal([64], stddev=0.01)),
			'convb5': tf.Variable(tf.random_normal([128], stddev=0.01)),
			'convb6': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'convb7': tf.Variable(tf.random_normal([128], stddev=0.01)),
			'convb8': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'convb9': tf.Variable(tf.random_normal([512], stddev=0.01)),
			'convb10': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'convb11': tf.Variable(tf.random_normal([512], stddev=0.01)),
			'convb12': tf.Variable(tf.random_normal([256], stddev=0.01)),
			'convb13': tf.Variable(tf.random_normal([512], stddev=0.01)),
			'convb14': tf.Variable(tf.random_normal([1024], stddev=0.01)),
			'convb15': tf.Variable(tf.random_normal([512], stddev=0.01)),
			'convb16': tf.Variable(tf.random_normal([1024], stddev=0.01)),
			'convb17': tf.Variable(tf.random_normal([512], stddev=0.01)),
			'convb18': tf.Variable(tf.random_normal([1024], stddev=0.01)),
			'convb19': tf.Variable(tf.random_normal([2], stddev=0.01)),
		}
		return weights, biases

	def buildNetBody(self, weights, biases):
		net = ConvLayer(self.inputImage, weights['convW1'], biases['convb1'], name='conv1')
		net = MaxPoolLayer(net, kernelSize=2, name='pool1')
		net = AlexNorm(inputTensor=net)

		net = ConvLayer(net, weights['convW2'], biases['convb2'], name='conv2')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')
		net = AlexNorm(inputTensor=net)

		net = ConvLayer(net, weights['convW3'], biases['convb3'], name='conv3')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW4'], biases['convb4'], name='conv4')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW5'], biases['convb5'], name='conv5')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')
		net = AlexNorm(inputTensor=net)

		net = ConvLayer(net, weights['convW6'], biases['convb6'], name='conv6')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW7'], biases['convb7'], name='conv7')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW8'], biases['convb8'], name='conv8')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')
		net = AlexNorm(inputTensor=net)


		net = ConvLayer(net, weights['convW9'], biases['convb9'], name='conv9')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW10'], biases['convb10'], name='conv10')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW11'], biases['convb11'], name='conv11')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW12'], biases['convb12'], name='conv12')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW13'], biases['convb13'], name='conv13')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')

		net = ConvLayer(net, weights['convW14'], biases['convb14'], name='conv14')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW15'], biases['convb15'], name='conv15')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW16'], biases['convb16'], name='conv16')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW17'], biases['convb17'], name='conv17')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW18'], biases['convb18'], name='conv18')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(net, weights['convW19'], biases['convb19'], name='conv19')
		net = AvgPoolLayer(net, kernelSize=3, name='poolFinal')
		net = MaxPoolLayer(net, kernelSize=5, name='poolFinal')
		output = tf.reshape(net, shape=[-1, 2])

		return output

