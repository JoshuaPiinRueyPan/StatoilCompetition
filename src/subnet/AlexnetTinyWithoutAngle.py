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
		net = ConvLayer(self.inputImage, 3, 8, stride_=1, padding_='SAME', layerName_='conv1')
		net = tf.nn.relu(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool1')
		net = AlexNorm(net, lsize=4, layerName_='norm1')

		net = ConvLayer(net, 3, 16, stride_=1, padding_='SAME', layerName_='conv2')
		net = tf.nn.relu(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')
		net = AlexNorm(net, lsize=4, layerName_='norm3')

		net = ConvLayer(net, 3, 16, stride_=1, padding_='SAME', layerName_='conv3')
		net = tf.nn.relu(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool3')
		net = AlexNorm(net, lsize=4, layerName_='norm3')

		net = ConvLayer(net, 3, 16, stride_=1, padding_='SAME', layerName_='conv4')
		net = tf.nn.relu(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool4')
		net = AlexNorm(net, lsize=4, layerName_='norm4')

		net = FullyConnectedLayer(net, numberOfOutputs_=128)
		net = tf.nn.relu(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer(net, numberOfOutputs_=128)
		net = tf.nn.relu(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		output = FullyConnectedLayer(net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)
		return output, tf.no_op()

