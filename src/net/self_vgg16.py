import tensorflow as tf
from src.net.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

VGG_INPUT_SIZE = [224, 224]

class self_vgg16(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.dropoutValue = 0.5

	def Build(self):
		vggInput = self.transformInputToVGG_Input()

		net = ConvLayer('Conv_1_1', vggInput, filterSize_=3, numberOfFilters_=64, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_1_1')
		net = ConvLayer('Conv_1_2', net, filterSize_=64, numberOfFilters_=64, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_1_2')
		net = MaxPoolLayer('Pool_1', net, kernelSize_=2, stride_=2, padding_='SAME')
			
		net = ConvLayer('Conv_2_1', net, filterSize_=64, numberOfFilters_=128, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_2_1')
		net = ConvLayer('Conv_2_2', net, filterSize_=128, numberOfFilters_=128, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_2_2')
		net = MaxPoolLayer('Pool_2', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = ConvLayer('Conv_3_1', net, filterSize_=128, numberOfFilters_=256, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_3_1')
		net = ConvLayer('Conv_3_2', net, filterSize_=256, numberOfFilters_=256, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_3_2')
		net = ConvLayer('Conv_3_3', net, filterSize_=256, numberOfFilters_=256, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_3_3')
		net = MaxPoolLayer('Pool_3', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = ConvLayer('Conv_4_1', net, filterSize_=256, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_4_1')
		net = ConvLayer('Conv_4_2', net, filterSize_=512, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_4_2')
		net = ConvLayer('Conv_4_3', net, filterSize_=512, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_4_3')
		net = MaxPoolLayer('Pool_4', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = ConvLayer('Conv_5_1', net, filterSize_=512, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_5_1')
		net = ConvLayer('Conv_5_2', net, filterSize_=512, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_5_2')
		net = ConvLayer('Conv_5_3', net, filterSize_=512, numberOfFilters_=512, stride_=1, padding_='SAME', isTrainable_=True)
		net = tf.nn.relu(net, 'RELU_5_3')
		net = MaxPoolLayer('Pool_5', net, kernelSize_=2, stride_=2, padding_='SAME')

		net = FullyConnectedLayer('Fc_6', net, numberOfOutputs_=4096)
		net = tf.nn.relu(net, 'RELU_6')
		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		net = FullyConnectedLayer('Fc_7', net, numberOfOutputs_=4096)
		net = tf.nn.relu(net, 'RELU_7')
		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

		logits = FullyConnectedLayer('Fc-Final', net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)

		return logits, tf.no_op()

	def transformInputToVGG_Input(self):
		with tf.name_scope("InputProcessing"):
			imagesHH = self.inputImage[:, :, :, 0]
			imagesHV = self.inputImage[:, :, :, 1]
			imagesAngle = tf.zeros_like( imagesHH )
			totalImages = tf.stack( [imagesHH, imagesHV, imagesAngle] )
			totalImages = tf.transpose(totalImages, [1, 2, 3, 0])
			totalImages = tf.image.resize_images(totalImages, VGG_INPUT_SIZE)

			return  totalImages
	
