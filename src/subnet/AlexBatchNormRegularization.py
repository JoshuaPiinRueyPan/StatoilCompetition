import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings
import settings.LayerSettings as layerSettings

slim = tf.contrib.slim

class AlexBatchNormRegularization(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_
		self.dropoutValue = 0.5

	def Build(self):
		batch_norm_params = {
			'decay': 0.997,
			'scale': True,
			'epsilon': 1e-5,
			'trainable': True,
			'is_training': True
		}


		'''
		with slim.arg_scope([slim.conv2d, slim.fc, slim.batch_norm, slim.dropout],
				    padding='SAME',
				    activation=None,
				    stddev=layerSettings.CONV_WEIGHTS_RNDOM_DEVIATION,
				    bias=0.0,
				    weights_regularizer=slim.l2_regularizer(0.0005),
				    normalizer_fn=slim.batch_norm,
				    normalizer_params=batch_norm_params,
				    trainable=True,
				    restore=True,
				    is_training=self.isTraining):
		'''
		with slim.arg_scope([slim.conv2d, slim.fully_connected],
				    weights_regularizer=slim.l2_regularizer(0.0005),
				    #normalizer_fn=slim.batch_norm,
				    #normalizer_params=batch_norm_params,
				    trainable=True):

			net = slim.conv2d(self.inputImage, 8, [3, 3], stride=1, padding='SAME', scope='Conv1')
			net = tf.nn.relu(net, name='RELU1')
			net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='MaxPool1')

			net = slim.conv2d(net, 16, [3, 3], stride=1, padding='SAME', scope='Conv2')
			net = tf.nn.relu(net, name='RELU2')
			net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='MaxPool2')

			net = slim.conv2d(net, 16, [3, 3], stride=1, padding='SAME', scope='Conv3')
			net = tf.nn.relu(net, name='RELU3')
			net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='MaxPool3')

			net = slim.conv2d(net, 16, [3, 3], stride=1, padding='SAME', scope='Conv4')
			net = tf.nn.relu(net, name='RELU4')
			net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='MaxPool4')

			net = slim.flatten(net, scope='flatten1')
			net = slim.fully_connected(net, num_outputs=128, scope='Fc1')
			net = tf.nn.relu(net, name='RELU5')

			net = slim.dropout(net, self.dropoutValue, scope='dropout1')
			net = slim.fully_connected(net, num_outputs=128, scope='Fc2')
			net = tf.nn.relu(net, name='RELU6')

			net = slim.dropout(net, self.dropoutValue, scope='dropout2')
			net = slim.fully_connected(net, num_outputs=outSettings.NUMBER_OF_CATEGORIES, scope='Fc3')


		listOfUpdateOperations = tf.get_collection(slim.ops.GraphKeys.UPDATE_OPS)
		return net, tf.group(*listOfUpdateOperations)

