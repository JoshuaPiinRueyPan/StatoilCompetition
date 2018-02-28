import tensorflow as tf
import numpy as np
import settings.LayerSettings as layerSettings
import settings.TrainingSettings as trainSettings
import os

def Create_tfVariable(variableName_, initialValue_, isTrainable_):
	tf_variable = tf.Variable(initialValue_, dtype=tf.float32, name=variableName_, trainable=isTrainable_)

	if tf_variable.get_shape() != initialValue_.get_shape():
		raise ValueError("In initialize variable: " + variableName_ + "\n" \
				 + "\t targetVariableShape = "+str(initialValue_.get_shape()) \
				 + ";  while actually create variable with shape = " + str(tf_variable.get_shape()) )

	return tf_variable


def CreateConvVariables(layerName_, filterSize_, inputChannels, numberOfFilters_, isTrainable_):
	with tf.name_scope(layerName_):
		weightsValue = tf.truncated_normal([filterSize_, filterSize_, inputChannels, numberOfFilters_],
						   mean=layerSettings.CONV_WEIGHTS_RNDOM_MEAN,
						   stddev=layerSettings.CONV_WEIGHTS_RNDOM_DEVIATION,
						   name="weightsValues")
		biasesValue = tf.truncated_normal([numberOfFilters_],
						  mean=layerSettings.CONV_BIASES_RNDOM_MEAN,
						  stddev=layerSettings.CONV_BIASES_RNDOM_DEVIATION,
						  name="biasesValues")
		weights = Create_tfVariable("weightsVariable", weightsValue, isTrainable_)
		biases = Create_tfVariable("biasesVariable", biasesValue, isTrainable_)
		return weights, biases


def CreateFcVariables(layerName_, numberOfInputs_, numberOfOutputs_, isTrainable_):
	with tf.name_scope(layerName_):
		weightsValue = tf.truncated_normal([numberOfInputs_, numberOfOutputs_],
						   mean=layerSettings.FC_WEIGHTS_RANDOM_MEAN,
						   stddev=layerSettings.FC_WEIGHTS_RANDOM_DEVIATION,
						   name="weightsValues")
		biasesValue = tf.truncated_normal([numberOfOutputs_],
						  mean=layerSettings.FC_BIASES_RANDOM_MEAN,
						  stddev=layerSettings.FC_BIASES_RANDOM_DEVIATION,
						  name="biasesValues")
		weights = Create_tfVariable("weightsVariable", weightsValue, isTrainable_)
		biases = Create_tfVariable("biasesVariable", biasesValue, isTrainable_)
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


