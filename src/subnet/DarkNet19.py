import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

class DarkNet19(SubnetBase):
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
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool1')

		net = ConvLayer(self.inputImage, 3, 64, stride_=1, padding_='SAME', layerName_='conv2')
		net, updateVariablesOp2 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')

		net = ConvLayer(self.inputImage, 3, 128, stride_=1, padding_='SAME', layerName_='conv3')
		net, updateVariablesOp3 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 64, stride_=1, padding_='SAME', layerName_='conv4')
		net, updateVariablesOp4 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 128, stride_=1, padding_='SAME', layerName_='conv5')
		net, updateVariablesOp5 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')

		net = ConvLayer(self.inputImage, 3, 256, stride_=1, padding_='SAME', layerName_='conv6')
		net, updateVariablesOp6 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 128, stride_=1, padding_='SAME', layerName_='conv7')
		net, updateVariablesOp7 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 256, stride_=1, padding_='SAME', layerName_='conv8')
		net, updateVariablesOp8 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')


		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv9')
		net, updateVariablesOp9 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 256, stride_=1, padding_='SAME', layerName_='conv10')
		net, updateVariablesOp10 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv11')
		net, updateVariablesOp11 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)

		net = ConvLayer(self.inputImage, 1, 256, stride_=1, padding_='SAME', layerName_='conv12')
		net, updateVariablesOp12 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 512, stride_=1, padding_='SAME', layerName_='conv13')
		net, updateVariablesOp13 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = MaxPoolLayer(net, kernelSize=2, layerName_='pool2')

		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv14')
		net, updateVariablesOp14 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 512, stride_=1, padding_='SAME', layerName_='conv15')
		net, updateVariablesOp15 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv16')
		net, updateVariablesOp16 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 512, stride_=1, padding_='SAME', layerName_='conv17')
		net, updateVariablesOp17 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 3, 1024, stride_=1, padding_='SAME', layerName_='conv18')
		net, updateVariablesOp18 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = ConvLayer(self.inputImage, 1, 2, stride_=1, padding_='SAME', layerName_='conv19')
		net, updateVariablesOp19 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = LeakyRELU(net)
		net = AvgPoolLayer(net, kernelSize=3, layerName_='poolFinal')
		net = MaxPoolLayer(net, kernelSize=5, layerName_='poolFinal')
		output = tf.reshape(net, shape=[-1, 2])

		updateVariablesOperations = tf.group(updateVariablesOp1, updateVariablesOp2, updateVariablesOp3,
						     updateVariablesOp4, updateVariablesOp5, updateVariablesOp6,
						     updateVariablesOp7, updateVariablesOp8, updateVariablesOp9,
						     updateVariablesOp10, updateVariablesOp11, updateVariablesOp12,
						     updateVariablesOp13, updateVariablesOp14, updateVariablesOp15,
						     updateVariablesOp16, updateVariablesOp17, updateVariablesOp18,
						     updateVariablesOp19)
		return output

