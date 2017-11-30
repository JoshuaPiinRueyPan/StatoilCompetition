import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class AlexnetTiny(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_
		self.dropoutValue = 0.5

	def Build(self):
		conv1 = ConvLayer(self.inputImage, 3, 8, stride_=1, padding_='SAME', layerName_='conv1')
		conv1 = tf.nn.relu(conv1)
		pool1 = MaxPoolLayer(conv1, kernelSize=2, layerName_='pool1')
		norm1 = AlexNorm(inputTensor=pool1)

		conv2 = ConvLayer(norm1, 3, 16, stride_=1, padding_='SAME', layerName_='conv2')
		conv2 = tf.nn.relu(conv2)
		pool2 = MaxPoolLayer(conv2, kernelSize=2, layerName_='pool1')
		norm2 = AlexNorm(inputTensor=pool2)

		conv3 = ConvLayer(norm2, 3, 16, stride_=1, padding_='SAME', layerName_='conv2')
		conv3 = tf.nn.relu(conv3)
		pool3 = MaxPoolLayer(conv3, kernelSize=2, layerName_='pool3')
		norm3 = AlexNorm(inputTensor=pool3)

		conv4 = ConvLayer(norm3, 3, 16, stride_=1, padding_='SAME', layerName_='conv2')
		conv4 = tf.nn.relu(conv4)
		pool4 = MaxPoolLayer(conv4, kernelSize=2, layerName_='pool4')
		norm4 = AlexNorm(inputTensor=pool4)

		pool4reshape = tf.reshape(norm4, [-1, 5*5*16])
		concat = tf.concat(axis=1, values=[pool4reshape, self.inputAngle])
		
		fc1 = FullyConnectedLayer(concat, numberOfOutputs_=128)
		fc1 = tf.nn.relu(fc1)
		fc1 = tf.cond(self.isTraining, lambda: tf.nn.dropout(fc1, self.dropoutValue), lambda: fc1)

		fc2 = FullyConnectedLayer(fc1, numberOfOutputs_=128)
		fc2 = tf.nn.relu(fc2)
		fc2 = tf.cond(self.isTraining, lambda: tf.nn.dropout(fc2, self.dropoutValue), lambda: fc2)

		output = FullyConnectedLayer(fc2, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)
		return output, tf.no_op()
