import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class AlexLargeBatchNorm(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_
		self.dropoutValue = 0.5

	def Build(self):
		net = ConvLayer(self.inputImage, 3, 64, stride_=1, padding_='SAME', layerName_='conv1')
		net, updateVariablesOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		#net = MaxPoolLayer(net, kernelSize=2, layerName_='pool1')

		net = ConvLayer(net, 3, 16, stride_=1, padding_='SAME', layerName_='conv2')
		net, updateVariablesOp2 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		#net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')

		net = ConvLayer(net, 3, 32, stride_=1, padding_='SAME', layerName_='conv3')
		net, updateVariablesOp3 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool3')

		net = ConvLayer(net, 3, 16, stride_=1, padding_='SAME', layerName_='conv4')
		net, updateVariablesOp4 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool4')

		net = FullyConnectedLayer(net, numberOfOutputs_=128, layerName_='fc1')
		net, updateVariablesOp5 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=False)
		net = LeakyRELU(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer(net, numberOfOutputs_=128, layerName_='fc2')
		net, updateVariablesOp6 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=False)
		net = LeakyRELU(net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer(net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES, layerName_='fc3')
		net, updateVariablesOp7 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=False)

		updateVariablesOperations = tf.group(updateVariablesOp1, updateVariablesOp2, updateVariablesOp3,
						    updateVariablesOp4, updateVariablesOp5, updateVariablesOp6,
						    updateVariablesOp7)
		return net, updateVariablesOperations

