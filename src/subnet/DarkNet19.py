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
		net = ConvLayer(self.inputImage, 3, 32, stride_=1, padding_='SAME', layerName_='conv1')
		net = MaxPoolLayer(net, kernelSize=2, name='pool1')
		net = AlexNorm(inputTensor=net)

		net = ConvLayer(self.inputImage, 3, 64, stride_=1, padding_='SAME', layerName_='conv2')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')
		net = AlexNorm(inputTensor=net)

		net = ConvLayer(self.inputImage, 3, 128, stride_=1, padding_='SAME', layerName_='conv3')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 1, 64, stride_=1, padding_='SAME', layerName_='conv4')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 3, 128, stride_=1, padding_='SAME', layerName_='conv5')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')
		net = AlexNorm(inputTensor=net)

		net = ConvLayer(self.inputImage, 3, 256, stride_=1, padding_='SAME', layerName_='conv6')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 1, 128, stride_=1, padding_='SAME', layerName_='conv7')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 3, 256, stride_=1, padding_='SAME', layerName_='conv8')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')
		net = AlexNorm(inputTensor=net)


		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv9')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 1, 256, stride_=1, padding_='SAME', layerName_='conv10')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv11')
		net = AlexNorm(inputTensor=net)

		net = ConvLayer(self.inputImage, 1, 256, stride_=1, padding_='SAME', layerName_='conv12')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv13')
		net = MaxPoolLayer(net, kernelSize=2, name='pool2')

		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv14')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 1, 512, stride_=1, padding_='SAME', layerName_='conv15')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv16')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 1, 512, stride_=1, padding_='SAME', layerName_='conv17')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv18')
		net = AlexNorm(inputTensor=net)
		net = ConvLayer(self.inputImage, 1, 2, stride_=1, padding_='SAME', layerName_='conv19')
		net = AvgPoolLayer(net, kernelSize=3, name='poolFinal')
		net = MaxPoolLayer(net, kernelSize=5, name='poolFinal')
		output = tf.reshape(net, shape=[-1, 2])

		return output

