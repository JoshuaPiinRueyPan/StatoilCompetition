import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
from src.layers.ResidualLayers import *
import settings.OutputSettings as outSettings

class ResnetFat3Fc(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_
		self.dropoutValue = 0.5

	def Build(self):
		with tf.variable_scope("Layer1"):
			net = ConvLayer(self.inputImage, filterSize_=5, numberOfFilters_=16,
					stride_=1, padding_='SAME', layerName_='conv1')
			net, updateOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
			net = LeakyRELU(net)
			net = MaxPoolLayer(net, kernelSize=2, layerName_='pool1')

		net, updateOp2 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=3, listOfConvFilterSize_=[8, 8, 32],
						activationType_="RELU", layerName_="Layer2")

		net, updateOp3 = ResidualLayer(self.isTraining, self.trainingStep, net,
						numberOfResidualBlocks_=4, listOfConvFilterSize_=[16, 16, 64],
						activationType_="RELU", layerName_="Layer3")

		net = MaxPoolLayer(net, kernelSize=5, layerName_="MaxPooling")

		net = FullyConnectedLayer(net, numberOfOutputs_=128, layerName_='Fc1')
		net, updateOp4 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=False)
		net = LeakyRELU(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer(net, numberOfOutputs_=128, layerName_='Fc2')

		net, updateOp5 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=False)
		net = LeakyRELU(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer(net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES, layerName_='Fc3')

		updateOperations = tf.group(updateOp1, updateOp2, updateOp3, updateOp4, updateOp5)

		return net, updateOperations

