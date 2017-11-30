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
		net = ConvLayer(self.inputImage, filterSize_=7, numberOfFilters_=8,
				stride_=1, padding_='SAME', layerName_='conv1')
		net, updateVariablesOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool1')

		net, updateOp1 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=3, listOfConvFilterSize_=[4, 4, 16],
						activationType_="RELU", layerName_="Layer2")

		net, updateOp2 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=4, listOfConvFilterSize_=[8, 8, 32],
						activationType_="RELU", layerName_="Layer3")

		net, updateOp3 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=6, listOfConvFilterSize_=[16, 16, 64],
						activationType_="RELU", layerName_="Layer3")

		net, updateOp4 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=3, listOfConvFilterSize_=[32, 32, 128],
						activationType_="RELU", layerName_="Layer3")

		net = AvgPoolLayer(net, kernelSize=7, layerName_="AveragePooling")

		net = FullyConnectedLayer(net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES, layerName_='Fc')

		updateOperations = tf.group(updateOp1, updateOp2, updateOp3, updateOp4)

		return net, updateOperations

