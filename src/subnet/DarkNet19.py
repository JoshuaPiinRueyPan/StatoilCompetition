import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.LayerHelper import *
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
		net = ConvLayer('conv1', self.inputImage, weights['convW1'], biases['convb1'])
		net = MaxPoolLayer('pool1', net, kernelSize=2)
		net = AlexNorm(inputTensor=net)

		net = ConvLayer('conv2', net, weights['convW2'], biases['convb2'])
		net = MaxPoolLayer('pool2', net, kernelSize=2)
		net = AlexNorm(inputTensor=net)

		net = ConvLayer('conv3', net, weights['convW3'], biases['convb3'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv4', net, weights['convW4'], biases['convb4'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv5', net, weights['convW5'], biases['convb5'])
		net = MaxPoolLayer('pool2', net, kernelSize=2)
		net = AlexNorm(inputTensor=net)

		net = ConvLayer('conv6', net, weights['convW6'], biases['convb6'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv7', net, weights['convW7'], biases['convb7'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv8', net, weights['convW8'], biases['convb8'])
		net = MaxPoolLayer('pool2', net, kernelSize=2)
		net = AlexNorm(inputTensor=net)


		net = ConvLayer('conv9', net, weights['convW9'], biases['convb9'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv10', net, weights['convW10'], biases['convb10'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv11', net, weights['convW11'], biases['convb11'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv12', net, weights['convW12'], biases['convb12'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv13', net, weights['convW13'], biases['convb13'])
		net = MaxPoolLayer('pool2', net, kernelSize=2)

		net = ConvLayer('conv14', net, weights['convW14'], biases['convb14'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv15', net, weights['convW15'], biases['convb15'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv16', net, weights['convW16'], biases['convb16'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv17', net, weights['convW17'], biases['convb17'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv18', net, weights['convW18'], biases['convb18'])
		net = AlexNorm(inputTensor=net)
		net = ConvLayer('conv19', net, weights['convW19'], biases['convb19'])
		net = AvgPoolLayer('poolFinal', net, kernelSize=3)
		net = MaxPoolLayer('poolFinal', net, kernelSize=5)
		output = tf.reshape(net, shape=[-1, 2])

		#output = tf.reshape(maxPoolFinal, shape=[-1, 2])
		
		return output

