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
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		net = ConvLayer(self.inputImage, 3, 32, stride_=1, padding_='SAME', layerName_='conv1')
		net, updateVariablesOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2)

		net = ConvLayer(net, 3, 64, stride_=1, padding_='SAME', layerName_='conv2')
		net, updateVariablesOp2 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2)

		net = ConvLayer(net, 3, 128, stride_=1, padding_='SAME', layerName_='conv3')
		net, updateVariablesOp3 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)

		net = ConvLayer(net, 3, 64, stride_=1, padding_='SAME', layerName_='conv4')
		net, updateVariablesOp4 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)

		net = ConvLayer(net, 3, 128, stride_=1, padding_='SAME', layerName_='conv5')
		net, updateVariablesOp5 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2)


		net = ConvLayer(net, 3, 256, stride_=1, padding_='SAME', layerName_='conv6')
		net, updateVariablesOp6 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)

		net = ConvLayer(net, 1, 2, stride_=1, padding_='SAME', layerName_='conv7')
		net, updateVariablesOp7 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)

		#net = AvgPoolLayer(net, kernelSize=10, layerName_='poolFinal')
		net = MaxPoolLayer(net, kernelSize=10, layerName_='poolFinal')
		output = tf.reshape(net, shape=[-1, 2])

		updateVariablesOperations = tf.group(updateVariablesOp1, updateVariablesOp2, updateVariablesOp3,
						    updateVariablesOp4, updateVariablesOp5, updateVariablesOp6,
						    updateVariablesOp7)
		return output, updateVariablesOperations

