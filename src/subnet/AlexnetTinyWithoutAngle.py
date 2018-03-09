import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class AlexnetTinyWithoutAngle(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_
		self.dropoutValue = 0.5

	def Build(self):
		net = ConvLayer('Conv1', self.inputImage, 3, 8, stride_=1, padding_='SAME')
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool1', net, kernelSize=2)
		net = AlexNorm('Norm1', net, lsize=4)

		net = ConvLayer('Conv2', net, 3, 16, stride_=1, padding_='SAME')
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool2', net, kernelSize=2)
		net = AlexNorm('Norm2', net, lsize=4)

		net = ConvLayer('Conv3', net, 3, 32, stride_=1, padding_='SAME')
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool3', net, kernelSize=2)
		net = AlexNorm('Norm3', net, lsize=4)

		net = ConvLayer('Conv4', net, 3, 32, stride_=1, padding_='SAME')
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool4', net, kernelSize=2)
		net = AlexNorm('Norm4', net, lsize=4)

		net = FullyConnectedLayer('Fc1', net, numberOfOutputs_=128)
		net = tf.nn.relu(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer('Fc2', net, numberOfOutputs_=128)
		net = tf.nn.relu(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		output = FullyConnectedLayer('Fc3', net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)
		return output, tf.no_op()

