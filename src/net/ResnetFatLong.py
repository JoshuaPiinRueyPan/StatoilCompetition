import tensorflow as tf
from src.net.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
from src.layers.ResidualLayers import *
import settings.OutputSettings as outSettings

class ResnetFatLong(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		with tf.name_scope('Layer1'):
			'''
			    Conv1st (5x5, 16) is the best result.
			    (5x5, 8) is also good.
			'''
			net = ConvLayer('Conv1', self.inputImage, filterSize_=5, numberOfFilters_=16,
					stride_=1, padding_='SAME')
			net, updateOp1 = BatchNormalization('BN_1', net, isConvLayer_=True,
							    isTraining_=self.isTraining, currentStep_=self.trainingStep)
			net = LeakyRELU('LeakyRELU', net)
			net = MaxPoolLayer('MaxPool1', net, kernelSize=2)

		# Layer 2
		net, updateOp2 = ResidualLayer( "ResLayer2",
						 net, numberOfResidualBlocks_=3, listOfConvFilterSize_=[8, 8, 32],
						 isTraining_=self.isTraining, trainingStep_=self.trainingStep, activationType_="RELU")

		# Layer 3
		net, updateOp3 = ResidualLayer( "ResLayer3",
						net, numberOfResidualBlocks_=4, listOfConvFilterSize_=[16, 16, 64],
						isTraining_=self.isTraining, trainingStep_=self.trainingStep, activationType_="RELU")

		# Layer 4
		net, updateOp4 = ResidualLayer( "ResLayer4",
						net, numberOfResidualBlocks_=4, listOfConvFilterSize_=[16, 16, 64],
						isTraining_=self.isTraining, trainingStep_=self.trainingStep, activationType_="RELU")

		# Layer 5
		net, updateOp5 = ResidualLayer( "ResLayer5",
						net, numberOfResidualBlocks_=4, listOfConvFilterSize_=[16, 16, 64],
						isTraining_=self.isTraining, trainingStep_=self.trainingStep, activationType_="RELU")

		'''
		    MaxPool seems a little improve (lower the loss).
		'''
		#net = AvgPoolLayer("AveragePool", net, kernelSize=38, padding_='VALID')
		net = AvgPoolLayer("AveragePool", net, kernelSize=7, padding_='SAME')
		#net = MaxPoolLayer("MaxPool", net, kernelSize=38, padding_='VALID')
		#net = MaxPoolLayer("MaxPool", net, kernelSize=7, padding_='SAME')

		net = FullyConnectedLayer('Fc', net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)

		updateOperations = tf.group(updateOp1, updateOp2, updateOp3, updateOp4, updateOp5, name="groupUpdateOps")

		return net, updateOperations

