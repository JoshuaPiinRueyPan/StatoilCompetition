import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class AllConv(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_
		self.dropoutValue = 0.5

	def Build(self):
		conv1 = ConvLayer(self.inputImage, 3, 8, stride_=1, padding_='SAME', layerName_='conv1')
		norm1 = AlexNorm(inputTensor=conv1)

		conv2 = ConvLayer(norm1, 3, 16, stride_=1, padding_='SAME', layerName_='conv2')
		norm2 = AlexNorm(inputTensor=conv2)

		conv3 = ConvLayer(norm2, 3, 32, stride_=1, padding_='SAME', layerName_='conv3')
		pool3 = MaxPoolLayer(conv3, kernelSize=2, layerName_='pool3')
		norm3 = AlexNorm(inputTensor=pool3)

		conv4 = ConvLayer(norm3, 3, 64, stride_=1, padding_='SAME', layerName_='conv4')
		pool4 = MaxPoolLayer(conv4, kernelSize=2, layerName_='pool4')
		norm4 = AlexNorm(inputTensor=pool4)

		conv5 = ConvLayer(norm4, 3, 128, stride_=1, padding_='SAME', layerName_='conv5')
		pool5 = MaxPoolLayer(conv5, kernelSize=2, layerName_='pool5')
		norm5 = AlexNorm(inputTensor=pool5)

		conv6 = ConvLayer(norm5, 3, 256, stride_=1, padding_='SAME', layerName_='conv6')
		pool6 = MaxPoolLayer(conv6, kernelSize=2, layerName_='pool6')
		norm6 = AlexNorm(inputTensor=pool6)

		conv7 = ConvLayer(norm6, 3, 256, stride_=1, padding_='SAME', layerName_='conv7')
		norm7 = AlexNorm(inputTensor=conv7)

		conv8 = ConvLayer(norm7, 3, 256, stride_=1, padding_='SAME', layerName_='conv8')

		avgpool = tf.nn.avg_pool(conv8, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID', name='avg_pool')
		avgpool = tf.reshape(avgpool, shape=[-1, 256])

		output = FullyConnectedLayer(avgpool, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)
		return output, tf.no_op()

