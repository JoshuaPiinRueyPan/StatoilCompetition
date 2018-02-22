import tensorflow as tf
import src.layers.LayerHelper as LayerHelper
import settings.LayerSettings as layerSettings

def ConvLayer(inputTensor_, filterSize_, numberOfFilters_, stride_=1, padding_='SAME', isTrainable_=True, layerName_=None):
	inputChannels = int(inputTensor_.shape[3])
	weights, biases = LayerHelper.CreateConvVariables(filterSize_, inputChannels, numberOfFilters_, isTrainable_, layerName_)
	convTensor = tf.nn.conv2d(inputTensor_, weights, strides=[1, stride_, stride_, 1], padding=padding_)
	outputTensor = tf.nn.bias_add(convTensor, biases)

	if layerName_ is not None:
		tf.summary.histogram(layerName_+'/weights', weights)
		tf.summary.histogram(layerName_+'/biases', biases)

	return outputTensor

def FullyConnectedLayer(inputTensor_, numberOfOutputs_, isTrainable_=True, layerName_=None):
	numberOfInputs = LayerHelper.CountElementsInOneFeatureMap(inputTensor_)
	inputTensor_ = tf.reshape(inputTensor_, [-1, numberOfInputs])
	weights, biases = LayerHelper.CreateFcVariables(numberOfInputs, numberOfOutputs_, isTrainable_, layerName_)
	fc = tf.matmul(inputTensor_, weights)
	output = tf.nn.bias_add(fc, biases)

	if layerName_ is not None:
		tf.summary.histogram(layerName_+'/weights', weights)
		tf.summary.histogram(layerName_+'/biases', biases)

	return output

def LeakyRELU(inputTensor_, layerName_="LeakyRELU"):
	return tf.maximum(layerSettings.LEAKY_RELU_FACTOR*inputTensor_, inputTensor_, name=layerName_)

def SetActivation(inputTensor_, activationType_):
	if activationType_.upper() == "RELU":
		return tf.nn.relu(inputTensor_)

	elif activationType_.upper() == "LEAKY_RELU":
		return LeakyRELU(inputTensor_)

	else:
		errorMessage = "You may add a new Activation type: '" + activationType_.upper() + "'\n"
		errorMessage += "but not implement in SetActivation()"
		raise ValueError(errorMessage)

def AlexNorm(inputTensor, lsize=4, layerName_=None):
	return tf.nn.lrn(inputTensor, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=layerName_)

def MaxPoolLayer(x, kernelSize=2, stride=2, layerName_=None):
	return tf.nn.max_pool(x, ksize=[1, kernelSize, kernelSize, 1],
				 strides=[1, stride, stride, 1], padding='VALID', name=layerName_)


def AvgPoolLayer(x, kernelSize=2, stride=2, layerName_=None):
	return tf.nn.avg_pool(x, ksize=[1, kernelSize, kernelSize, 1],
				 strides=[1, stride, stride, 1],
				 padding='VALID',
				 name=layerName_)

def BatchNormalization(isTraining_, currentStep_, inputTensor_, isConvLayer_=False, isTrainable_=True, layerName_="BatchNorm"):
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
		gamma = LayerHelper.variableManager.LoadOrCreateVariable( tf.ones([outputChannels]), isTrainable_, layerName_+"_gamma" )
		betta = LayerHelper.variableManager.LoadOrCreateVariable( tf.zeros([outputChannels]), isTrainable_, layerName_+"_betta" )
		epsilon = 1e-5
		outputTensor = tf.nn.batch_normalization(inputTensor_, mean=totalMean, variance=totalVariance, offset=betta,
							 scale=gamma, variance_epsilon=epsilon)
		return outputTensor, updateVariablesOperation
