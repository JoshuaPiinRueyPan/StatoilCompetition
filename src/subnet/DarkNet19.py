import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

DARKNET_INPUT_SIZE = [416, 416]

class DarkNet19(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		darknetInput = self.transformInput()
		print("input = " + str(darknetInput.get_shape()) )

		net = ConvLayer(darknetInput, 3, 32, stride_=1, padding_='SAME', layerName_='conv1')
		net, updateVariablesOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN1")
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool1')

		net = ConvLayer(self.inputImage, 3, 64, stride_=1, padding_='SAME', layerName_='conv2')
		net, updateVariablesOp2 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN2")
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')

		net = ConvLayer(self.inputImage, 3, 128, stride_=1, padding_='SAME', layerName_='conv3')
		net, updateVariablesOp3 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN3")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 64, stride_=1, padding_='SAME', layerName_='conv4')
		net, updateVariablesOp4 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN4")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 128, stride_=1, padding_='SAME', layerName_='conv5')
		net, updateVariablesOp5 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN5")
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')

		net = ConvLayer(self.inputImage, 3, 256, stride_=1, padding_='SAME', layerName_='conv6')
		net, updateVariablesOp6 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN6")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 128, stride_=1, padding_='SAME', layerName_='conv7')
		net, updateVariablesOp7 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN7")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 256, stride_=1, padding_='SAME', layerName_='conv8')
		net, updateVariablesOp8 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN8")
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')


		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv9')
		net, updateVariablesOp9 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN9")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 256, stride_=1, padding_='SAME', layerName_='conv10')
		net, updateVariablesOp10 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN10")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv11')
		net, updateVariablesOp11 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN11")
		net = LeakyRELU(net)

		net = ConvLayer(self.inputImage, 1, 256, stride_=1, padding_='SAME', layerName_='conv12')
		net, updateVariablesOp12 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN12")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv13')
		net, updateVariablesOp13 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN13")
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')

		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv14')
		net, updateVariablesOp14 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN14")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 512, stride_=1, padding_='SAME', layerName_='conv15')
		net, updateVariablesOp15 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN15")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv16')
		net, updateVariablesOp16 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN16")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 512, stride_=1, padding_='SAME', layerName_='conv17')
		net, updateVariablesOp17 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN17")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv18')
		net, updateVariablesOp18 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN18")
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 2, stride_=1, padding_='SAME', layerName_='conv19')
		net, updateVariablesOp19 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True, layerName_="BN19")
		net = LeakyRELU(net)

		print("last.shape() = " + str(net.get_shape()) )
		net = AvgPoolLayer(net, kernelSize=75, stride=1, layerName_='poolFinal')
		print("after AvgPool, shape() = " + str(net.get_shape()))
		output = FullyConnectedLayer(net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)

		updateVariablesOperations = tf.group(updateVariablesOp1, updateVariablesOp2, updateVariablesOp3,
						     updateVariablesOp4, updateVariablesOp5, updateVariablesOp6,
						     updateVariablesOp7, updateVariablesOp8, updateVariablesOp9,
						     updateVariablesOp10, updateVariablesOp11, updateVariablesOp12,
						     updateVariablesOp13, updateVariablesOp14, updateVariablesOp15,
						     updateVariablesOp16, updateVariablesOp17, updateVariablesOp18,
						     updateVariablesOp19)
		return output, updateVariablesOperations

	def transformInput(self):
		imagesHH = self.inputImage[:, :, :, 0]
		imagesHV = self.inputImage[:, :, :, 1]
		imagesAngle = tf.zeros_like( imagesHH )
		totalImages = tf.stack( [imagesHH, imagesHV, imagesAngle] )
		totalImages = tf.transpose(totalImages, [1, 2, 3, 0])
		totalImages = tf.image.resize_images(totalImages, DARKNET_INPUT_SIZE)
		
		return  totalImages
		
