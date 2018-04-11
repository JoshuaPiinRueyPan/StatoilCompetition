import tensorflow as tf
from src.net.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class AlexnetBatchNorm(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_
		self.dropoutValue = 0.5

	def Build(self):
		net = ConvLayer('Conv1', self.inputImage, 3, 8, stride_=1, padding_='SAME')
		net, updateVariablesOp1 = BatchNormalization('BN1', net, isConvLayer_=True,
							     isTraining_=self.isTraining, currentStep_=self.trainingStep)
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool1', net, kernelSize_=2)

		net = ConvLayer('Conv2', net, 3, 16, stride_=1, padding_='SAME')
		net, updateVariablesOp2 = BatchNormalization('BN2', net, isConvLayer_=True,
							     isTraining_=self.isTraining, currentStep_=self.trainingStep)
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool2', net, kernelSize_=2)

		net = ConvLayer('Conv3', net, 3, 16, stride_=1, padding_='SAME')
		net, updateVariablesOp3 = BatchNormalization('BN3', net, isConvLayer_=True, 
							     isTraining_=self.isTraining, currentStep_=self.trainingStep)
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool3', net, kernelSize_=2)

		net = ConvLayer('Conv4', net, 3, 16, stride_=1, padding_='SAME')
		net, updateVariablesOp4 = BatchNormalization('BN4', net, isConvLayer_=True,
							     isTraining_=self.isTraining, currentStep_=self.trainingStep)
		net = tf.nn.relu(net)
		net = MaxPoolLayer('Pool4', net, kernelSize_=2)

		net = FullyConnectedLayer('Fc1', net, numberOfOutputs_=128)
		net, updateVariablesOp1 = BatchNormalization('BN5', net, isConvLayer_=False,
							     isTraining_=self.isTraining, currentStep_=self.trainingStep)
		net = LeakyRELU('LeakyRELU1', net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer('Fc2', net, numberOfOutputs_=128)
		net, updateVariablesOp2 = BatchNormalization('BN6', net, isConvLayer_=False,
							     isTraining_=self.isTraining, currentStep_=self.trainingStep)
		net = LeakyRELU('LeakyRELU2', net)

		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer('Fc3', net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)

		updateVariablesOperations = tf.group(updateVariablesOp1, updateVariablesOp2)
		return net, updateVariablesOperations

