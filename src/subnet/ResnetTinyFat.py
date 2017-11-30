import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
from src.layers.ResidualLayers import *
import settings.OutputSettings as outSettings

class ResnetTiny(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		with tf.variable_scope("Layer1"):
			'''
			    Conv1st (5x5, 16) is the best result.
			    (5x5, 8) is also good.
			'''
			net = ConvLayer(self.inputImage, filterSize_=5, numberOfFilters_=16,
					stride_=1, padding_='SAME', layerName_='conv1')
			net, updateVariablesOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
			net = LeakyRELU(net)
			net = MaxPoolLayer(net, kernelSize=2, layerName_='pool1')

		net, updateOp1 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=3, listOfConvFilterSize_=[8, 8, 32],
						activationType_="RELU", layerName_="Layer2")

		net, updateOp2 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=4, listOfConvFilterSize_=[16, 16, 64],
						activationType_="RELU", layerName_="Layer3")

		'''
		    MaxPool seems a little improve (lower the loss).
		'''
		#net = AvgPoolLayer(net, kernelSize=7, layerName_="AveragePooling")
		net = MaxPoolLayer(net, kernelSize=5, layerName_="MaxPooling")

		net = FullyConnectedLayer(net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES, layerName_='Fc')

		updateOperations = tf.group(updateOp1, updateOp2, updateOp3, updateOp4)

		return net, updateOperations

