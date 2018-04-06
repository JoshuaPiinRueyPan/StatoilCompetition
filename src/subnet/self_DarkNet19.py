import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

VGG_INPUT_SIZE = [224, 224]
DARKNET19_MODEL_PATH = 'data/darknet19/darknet19.pb'

class self_DarkNet19(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_

	def Build(self):
		darknetInput = self.transformInputToVGG_Input()

		net = ConvLayer('Conv1', darknetInput, filterSize_=3, numberOfFilters_=32, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp1 = BatchNormalization('BN1', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU1', net)
		net = MaxPoolLayer('Pool1', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = ConvLayer('Conv2', net, filterSize_=3, numberOfFilters_=64, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp2 = BatchNormalization('BN2', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU2', net)
		net = MaxPoolLayer('Pool2', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = ConvLayer('Conv3', net, 3, numberOfFilters_=128, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp3 = BatchNormalization('BN3', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU3', net)
		net = ConvLayer('Conv4', net, 1, numberOfFilters_=64, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp4 = BatchNormalization('BN4', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU4', net)
		net = ConvLayer('Conv5', net, 3, numberOfFilters_=128, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp5 = BatchNormalization('BN5', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU5', net)
		net = MaxPoolLayer('Pool5', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = ConvLayer('Conv6', net, 3, numberOfFilters_=256, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp6 = BatchNormalization('BN6', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU6', net)
		net = ConvLayer('Conv7', net, 1, numberOfFilters_=128, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp7 = BatchNormalization('BN7', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU7', net)
		net = ConvLayer('Conv8', net, 3, numberOfFilters_=256, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp8 = BatchNormalization('BN8', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU8', net)
		net = MaxPoolLayer('Pool8', net, kernelSize_=2, stride_=2, padding_='SAME')


		net = ConvLayer('Conv9', net, 3, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp9 = BatchNormalization('BN9', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU9', net)
		net = ConvLayer('Conv10', net, 1, numberOfFilters_=256, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp10 = BatchNormalization('BN10', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU10', net)
		net = ConvLayer('Conv11', net, 3, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp11 = BatchNormalization('BN11', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU11', net)

		net = ConvLayer('Conv12', net, 1, numberOfFilters_=256, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp12 = BatchNormalization('BN12', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU12', net)
		net = ConvLayer('Conv13', net, 3, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp13 = BatchNormalization('BN13', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU13', net)
		net = MaxPoolLayer('Pool13', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = ConvLayer('Conv14', net, 3, numberOfFilters_=1024, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp14 = BatchNormalization('BN14', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU14', net)
		net = ConvLayer('Conv15', net, 1, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp15 = BatchNormalization('BN15', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU15', net)
		net = ConvLayer('Conv16', net, 3, numberOfFilters_=1024, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp16 = BatchNormalization('BN16', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU16', net)
		net = ConvLayer('Conv17', net, 1, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp17 = BatchNormalization('BN17', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU17', net)
		net = ConvLayer('Conv18', net, 3, numberOfFilters_=1024, stride_=1, padding_='SAME', isTrainable_=True)
		net, updateVariablesOp18 = BatchNormalization('BN18', net, isConvLayer_=True, isTraining_=self.isTraining,
							     currentStep_=self.trainingStep, isTrainable_=True)
		net = LeakyRELU('RELU18', net)
		net = ConvLayer('Conv19', net, 1, numberOfFilters_=outSettings.NUMBER_OF_CATEGORIES,
				 stride_=1, padding_='SAME', isTrainable_=True )

		net = AvgPoolLayer('AvgPool', net, kernelSize_=7, stride_=1, padding_='VALID')
		logits = tf.reshape(net, [-1, outSettings.NUMBER_OF_CATEGORIES])

		updateVariablesOperations = tf.group(updateVariablesOp1, updateVariablesOp2, updateVariablesOp3,
						     updateVariablesOp4, updateVariablesOp5, updateVariablesOp6,
						     updateVariablesOp7, updateVariablesOp8, updateVariablesOp9,
						     updateVariablesOp10, updateVariablesOp11, updateVariablesOp12,
						     updateVariablesOp13, updateVariablesOp14, updateVariablesOp15,
						     updateVariablesOp16, updateVariablesOp17, updateVariablesOp18)

		return logits, updateVariablesOperations

	def transformInputToVGG_Input(self):
		with tf.name_scope("InputProcessing"):
			imagesHH = self.inputImage[:, :, :, 0]
			imagesHV = self.inputImage[:, :, :, 1]
			imagesAngle = tf.zeros_like( imagesHH )
			totalImages = tf.stack( [imagesHH, imagesHV, imagesAngle] )
			totalImages = tf.transpose(totalImages, [1, 2, 3, 0])
			totalImages = tf.image.resize_images(totalImages, VGG_INPUT_SIZE)
			
			return  totalImages
	
