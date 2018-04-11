import tensorflow as tf
from src.net.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
from src.layers.ResidualLayers import *
import settings.OutputSettings as outSettings

class ResnetLong(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		with tf.variable_scope("Layer1"):
			net = ConvLayer(self.inputImage, filterSize_=5, numberOfFilters_=8,
					stride_=1, padding_='SAME', name_='conv1')
			net, updateOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
			net = LeakyRELU(net)
			net = MaxPoolLayer(net, kernelSize=2, name_='pool1')

		net, updateOp2 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=3, listOfConvFilterSize_=[2, 2, 4],
						activationType_="RELU", name_="Layer2")

		net, updateOp3 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=4, listOfConvFilterSize_=[2, 2, 4],
						activationType_="RELU", name_="Layer3")

		net, updateOp4 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=6, listOfConvFilterSize_=[2, 2, 4],
						activationType_="RELU", name_="Layer4")

		net, updateOp5 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=3, listOfConvFilterSize_=[4, 4, 8],
						activationType_="RELU", name_="Layer5")

		net = MaxPoolLayer(net, kernelSize=5, name_="MaxPooling")
		#net = AvgPoolLayer(net, kernelSize=5, name_="AvgPooling")

		net = FullyConnectedLayer(net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES, name_='Fc')

		updateOperations = tf.group(updateOp1, updateOp2, updateOp3, updateOp4, updateOp5)

		return net, updateOperations

