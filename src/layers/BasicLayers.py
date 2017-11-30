import tensorflow as tf
from src.layers.LayerHelper import *
import settings.LayerSettings as layerSettings

def ConvLayer(inputTensor_, filterSize_, numberOfFilters_, stride_=1, padding_='SAME', layerName_=None):
	inputChannels = int(inputTensor_.shape[3])
	weights, biases = CreateConvVariables(filterSize_, inputChannels, numberOfFilters_, layerName_)
	convTensor = tf.nn.conv2d(inputTensor_, weights, strides=[1, stride_, stride_, 1], padding=padding_)
	outputTensor = tf.nn.bias_add(convTensor, biases)

	if layerName_ is not None:
		tf.summary.histogram(layerName_+'/weights', weights)
		tf.summary.histogram(layerName_+'/biases', biases)

	return outputTensor

def FullyConnectedLayer(inputTensor_, numberOfOutputs_, layerName_=None):
	numberOfInputs = CountElementsInOneFeatureMap(inputTensor_)
	inputTensor_ = tf.reshape(inputTensor_, [-1, numberOfInputs])
	weights, biases = CreateFcVariables(numberOfInputs, numberOfOutputs_, layerName_)
	fc = tf.matmul(inputTensor_, weights)
	output = tf.nn.bias_add(fc, biases)

	if layerName_ is not None:
		tf.summary.histogram(layerName_+'/weights', weights)
		tf.summary.histogram(layerName_+'/biases', biases)

	return output

def LeakyRELU(inputTensor_, layerName_="LeakyRELU"):
	return tf.maximum(layerSettings.LEAKY_RELU_FACTOR*inputTensor_, inputTensor_, name=layerName_)

def AlexNorm(inputTensor, lsize=4, layerName_=None):
	return tf.nn.lrn(inputTensor, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=layerName_)

def MaxPoolLayer(x, kernelSize=2, layerName_=None):
	return tf.nn.max_pool(x, ksize=[1, kernelSize, kernelSize, 1],
				 strides=[1, kernelSize, kernelSize, 1], padding='SAME', name=layerName_)


def AvgPoolLayer(x, kernelSize, layerName_=None):
	return tf.nn.avg_pool(x, ksize=[1, kernelSize, kernelSize, 1],
				 strides=[1, kernelSize, kernelSize, 1],
				 padding='SAME',
				 name=layerName_)

def BatchNormalization(isTraining_, currentStep_, inputTensor_, isConvLayer_=False, layerName_="BatchNorm"):
	with tf.variable_scope(layerName_):
		currentBatchMean = None
		currentBatchVariance = None
		outputChannels = None
		if isConvLayer_:
			currentBatchMean, currentBatchVariance = tf.nn.moments(inputTensor_, [0, 1, 2])
		else:
			currentBatchMean, currentBatchVariance = tf.nn.moments(inputTensor_, [0])

		'''
		    The mean & variance of variables in Testing will be calculated by following methods:
			decay = min(decay, (1 + step)/(10+step) )
			totalMean = (1 - decay)*currentBatchMean + decay*totalMean
		    in which totalMean also called 'shadow variable' of currentBatchMean.

		    Note: It's meaningless to take the first step (which has been Random Initialized)
		    for average.  Therefore, one should add the training step to scale the previous
		    steps down.
		'''
		averageCalculator = tf.train.ExponentialMovingAverage(decay=layerSettings.BATCH_NORMALIZATION_MOVING_AVERAGE_DECAY_RATIO,
								      num_updates=currentStep_)
		updateVariablesOperation = averageCalculator.apply( [currentBatchMean, currentBatchVariance] )

		totalMean = tf.cond(isTraining_,
				    lambda: currentBatchMean, lambda: averageCalculator.average(currentBatchMean) )

		totalVariance = tf.cond(isTraining_,
					lambda: currentBatchVariance, lambda: averageCalculator.average(currentBatchVariance) )
		'''
		    Following will calculate BatchNormalization by:
			X' = gamma * (X - mean)/sqrt(variance**2 + epsillon) + betta
		'''
		outputChannels = int(inputTensor_.shape[-1])
		gamma = tf.Variable( tf.ones([outputChannels]) )
		betta = tf.Variable( tf.zeros([outputChannels]) )
		epsilon = 1e-5
		outputTensor = tf.nn.batch_normalization(inputTensor_, mean=totalMean, variance=totalVariance, offset=betta,
							 scale=gamma, variance_epsilon=epsilon)
		return outputTensor, updateVariablesOperation
