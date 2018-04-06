import numpy as np
import tensorflow as tf
from src.layers.BasicLayers import *

from src.subnet.SubnetBase import SubnetBase

VGG_MODEL_PATH = "data/VGG/vgg16.npy"

VGG_INPUT_SIZE = [224, 224]

class VGG16BatchNorm(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

		try:
			self.data_dict = np.load(VGG_MODEL_PATH, encoding='latin1').item()
			print("Load VGG pretrain model...")

		except:
			self.data_dict = None
			print("No VGG16 pretrain model found!  If you want to use the pretrain model, please")
			print("download vgg16.npy from: https://github.com/machrisaa/tensorflow-vgg")


	def Build(self):
		vggInput = self.transformInputToVGG_Input()

		net = self.conv_layer(vggInput, 3, 64, "conv1_1")
		net, updateOp1 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.conv_layer(net, 64, 64, "conv1_2")
		net, updateOp2 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.max_pool(net, 'pool1')

		net = self.conv_layer(net, 64, 128, "conv2_1")
		net, updateOp3 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.conv_layer(net, 128, 128, "conv2_2")
		net, updateOp4 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.max_pool(net, 'pool2')

		net = self.conv_layer(net, 128, 256, "conv3_1")
		net, updateOp5 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.conv_layer(net, 256, 256, "conv3_2")
		net, updateOp6 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.conv_layer(net, 256, 256, "conv3_3")
		net, updateOp7 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.max_pool(net, 'pool3')

		net = self.conv_layer(net, 256, 512, "conv4_1")
		net, updateOp8 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.conv_layer(net, 512, 512, "conv4_2")
		net, updateOp9 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.conv_layer(net, 512, 512, "conv4_3")
		net, updateOp10 = BatchNormalization(self.isTraining, self.trainingStep, net, isConvLayer_=True)
		net = self.max_pool(net, 'pool4')

		net = self.conv_layer(net, 512, 512, "conv5_1")
		net = self.conv_layer(net, 512, 512, "conv5_2")
		net = self.conv_layer(net, 512, 512, "conv5_3")
		net = self.max_pool(net, 'pool5')

		net = self.fc_layer(net, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
		net = tf.nn.relu(net)
		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, 0.5), lambda: net)

		net = self.fc_layer(net, 4096, 4096, "fc7")
		net = tf.nn.relu(net)
		net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, 0.5), lambda: net)

		fcFinal = self.fc_layer(net, 4096, 2, "fcFinal")

		updateOperations = tf.group(updateOp1, updateOp2, updateOp3, updateOp4, updateOp5,
					    updateOp6, updateOp7, updateOp8, updateOp9, updateOp10)
		self.data_dict = None
		return fcFinal, updateOperations


	def transformInputToVGG_Input(self):
		imagesHH = self.inputImage[:, :, :, 0]
		imagesHV = self.inputImage[:, :, :, 1]
		imagesAngle = tf.zeros_like( imagesHH )
		totalImages = tf.stack( [imagesHH, imagesHV, imagesAngle] )
		totalImages = tf.transpose(totalImages, [1, 2, 3, 0])
		totalImages = tf.image.resize_images(totalImages, VGG_INPUT_SIZE)
		
		return  totalImages
		

	def conv_layer(self, bottom, in_channels, out_channels, name):
		with tf.variable_scope(name):
			filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

			conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
			bias = tf.nn.bias_add(conv, conv_biases)
			relu = tf.nn.relu(bias)

			return relu

	def max_pool(self, bottom, name):
		return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	def fc_layer(self, bottom, in_size, out_size, name):
		with tf.variable_scope(name):
			weights, biases = self.get_fc_var(in_size, out_size, name)

			x = tf.reshape(bottom, [-1, in_size])
			fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

			return fc

	def get_conv_var(self, filter_size, in_channels, out_channels, name):
		initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
		filters = self.get_var(initial_value, name, 0, name + "_filters")

		initial_value = tf.truncated_normal([out_channels], .0, .001)
		biases = self.get_var(initial_value, name, 1, name + "_biases")

		return filters, biases

	def get_fc_var(self, in_size, out_size, name):
		initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
		weights = self.get_var(initial_value, name, 0, name + "_weights")

		initial_value = tf.truncated_normal([out_size], .0, .001)
		biases = self.get_var(initial_value, name, 1, name + "_biases")

		return weights, biases

	def get_var(self, initial_value, name, idx, var_name):
		if self.data_dict is not None and name in self.data_dict:
			value = self.data_dict[name][idx]
		else:
			value = initial_value

		var = tf.Variable(value, name=var_name)

		assert var.get_shape() == initial_value.get_shape()

		return var

