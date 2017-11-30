import tensorflow as tf
import numpy as np
import settings.LayerSettings as layerSettings


def CreateConvVariables(filterSize_, inputChannels, numberOfFilters_, layerName_):
	weightsValue = tf.truncated_normal([filterSize_, filterSize_, inputChannels, numberOfFilters_],
					   mean=layerSettings.CONV_WEIGHTS_RNDOM_MEAN,
					   stddev=layerSettings.CONV_WEIGHTS_RNDOM_DEVIATION)
	biasesValue = tf.truncated_normal([numberOfFilters_],
					  mean=layerSettings.CONV_BIASES_RNDOM_MEAN,
					  stddev=layerSettings.CONV_BIASES_RNDOM_DEVIATION)
	if layerName_ is None:
		weights = tf.Variable(weightsValue)
		biases = tf.Variable(biasesValue)
		return weights, biases

	else:
		weights = tf.Variable(weightsValue, layerName_ + "_weightings")
		biases = tf.Variable(biasesValue, layerName_ + "_biases")
		return weights, biases


def CreateFcVariables(numberOfInputs_, numberOfOutputs_, layerName_):
	weightsValue = tf.truncated_normal([numberOfInputs_, numberOfOutputs_],
					   mean=layerSettings.FC_WEIGHTS_RANDOM_MEAN,
					   stddev=layerSettings.FC_WEIGHTS_RANDOM_DEVIATION)
	biasesValue = tf.truncated_normal([numberOfOutputs_],
					  mean=layerSettings.FC_BIASES_RANDOM_MEAN,
					  stddev=layerSettings.FC_BIASES_RANDOM_DEVIATION)
	if layerName_ is None:
		weights = tf.Variable(weightsValue)
		biases = tf.Variable(biasesValue)
		return weights, biases
	else:
		weights = tf.Variable(weightsValue, layerName_ + "_weightings")
		biases = tf.Variable(biasesValue, layerName_ + "_biases")
		return weights, biases


def CountElementsInOneFeatureMap(inputTensor_):
	'''
	   This function calculate number of elements in an image.
	   For example, if you have a feature map with (b, w, h, c)
	   this function will return w*h*c.  i.e. without consider
	   the batch dimension.
	'''
	featureMapShape = inputTensor_.shape[1:]
	return int( np.prod(featureMapShape) )


