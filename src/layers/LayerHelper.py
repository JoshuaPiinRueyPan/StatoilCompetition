import tensorflow as tf
import numpy as np
import settings.LayerSettings as layerSettings
import settings.TrainingSettings as trainSettings
import os

'''
    This Class control the variables creation, save, & recover.
    The save and recover features are implemented by using *.npy format.
    Note: The layer-by-layer-training can be done by the virtue of NPY format.
'''
class VariablesManager:
	def RecoverDictionaryFromCheckpoint(self, PATH_TO_NPY_FORMAT_CHECKPOINT_):
		if os.path.exists(PATH_TO_NPY_FORMAT_CHECKPOINT_):
			print("Load pretrain model from npy file: " + PATH_TO_NPY_FORMAT_CHECKPOINT_)
			self._dictionaryFromCheckpoint = np.load(PATH_TO_NPY_FORMAT_CHECKPOINT_, encoding='latin1').item()

		else:
			raise ValueError("Pretrain model: " + PATH_TO_NPY_FORMAT_CHECKPOINT_ + " not found!")

	def LoadOrCreateVariable(self, initialValue_, isTrainable_, variableName_):
		variableValue = initialValue_
		try:
			variableValue = self._loadVariableFromCheckpoint(variableName_)
			print(variableName_ + ": init from dictionary")

		except TypeError:
			print(variableName_ + ": init from scratch")
			variableValue = initialValue_

		except KeyError:
			print(variableName_ + ": key not found!")
			variableValue = initialValue_

		if isTrainable_:
			tf_variable = tf.Variable(variableValue, dtype=tf.float32, name=variableName_)

		else:
			tf_variable = tf.constant(variableValue, dtype=tf.float32, name=variableName_)

		self._appendVariableToDictionary(variableName_, tf_variable)

		if tf_variable.get_shape() != initialValue_.get_shape():
			raise ValueError("In initialize variable: " + variableName_ + "\n" \
					 + "\t targetVariableShape = "+str(initialValue_.get_shape()) \
					 + ";  while actually create variable with shape = " + str(tf_variable.get_shape()) )


		if variableName_ == 'Conv2/weightsTensor':
			print('Conv2/w[1, 0, 1, :] = ')
			print('\t In dictionary, = \n' + str(variableValue[1, 0, 1, :]))
			self.tempConvVariable = tf_variable

			with tf.Session() as sess:
				sess = tf.Session()
				init = tf.initialize_all_variables()
				sess.run(init)
				result = sess.run(self.tempConvVariable)
				print("after sess.run(variable) = \n" + str(result[1, 0, 1, :]))
			print("tempConvVariable = " + str(self.tempConvVariable))

		if variableName_ == 'BN3/Gamma':
			print('BN3/Gamma = ')
			print('In dict = \n' + str(variableValue))
			self.tempBN = tf_variable

			with tf.Session() as sess:
				sess = tf.Session()
				init = tf.initialize_all_variables()
				sess.run(init)
				result = sess.run(self.tempBN)
				print("after sess.run(variable) = \n" + str(result))
			print("tempBN = " + str(self.tempBN))

		return tf_variable

	def _loadVariableFromCheckpoint(self, variableName_):
		return self._dictionaryFromCheckpoint[variableName_]

	def _appendVariableToDictionary(self, variableName_, variableValue_):
		self._dictionaryInCurrentNetwork[variableName_] = variableValue_

	def SaveAllNetworkVariables(self, tf_session, FILE_PATH_NAME_TO_SAVE_VARIABLES):
		tempDictionary = {}
		for variableName, tf_variable in self._dictionaryInCurrentNetwork.items():
			if variableName in tempDictionary:
				raise ValueError("The same variable: '" + variableName + "' exists!\n" \
						 + "\t Please check if all of your layer name are different.")

			else:
				variableValue = tf_session.run(tf_variable)
				tempDictionary[variableName] = variableValue

		np.save(FILE_PATH_NAME_TO_SAVE_VARIABLES, tempDictionary)
		print('Conv2/w[1, 0, 1, :] = ')
		print('\t In tempDict, = \n' + str(tempDictionary['Conv2/weightsTensor'][1, 0, 1, :]))
		print('BN3/Gamma = ')
		print('\t In tempDict, = \n' + str(tempDictionary['BN3/Gamma']))


	def __init__(self):
		CHECKPOINT_PATH_FILE_NAME, CHECKPOINT_FILE_TYPE = os.path.splitext(trainSettings.PRETRAIN_MODEL_PATH_NAME)
		if CHECKPOINT_FILE_TYPE == ".npy":
			'''
			    Following is a dictionary that map 'variableName' to 'variableValue'
			'''
			self.RecoverDictionaryFromCheckpoint(trainSettings.PRETRAIN_MODEL_PATH_NAME)

		else:
			self._dictionaryFromCheckpoint = None

		'''
		    Following is a dictionary that map 'variableName' to 'variableTensor'
		'''
		self._dictionaryInCurrentNetwork = {}



variableManager = VariablesManager()
def CreateConvVariables(filterSize_, inputChannels, numberOfFilters_, isTrainable_, layerName_):
	weightsValue = tf.truncated_normal([filterSize_, filterSize_, inputChannels, numberOfFilters_],
					   mean=layerSettings.CONV_WEIGHTS_RNDOM_MEAN,
					   stddev=layerSettings.CONV_WEIGHTS_RNDOM_DEVIATION,
					   name=layerName_+"/createWeightsValues")
	biasesValue = tf.truncated_normal([numberOfFilters_],
					  mean=layerSettings.CONV_BIASES_RNDOM_MEAN,
					  stddev=layerSettings.CONV_BIASES_RNDOM_DEVIATION,
					  name=layerName_+"/createBiasesValues")
	weights = variableManager.LoadOrCreateVariable(weightsValue, isTrainable_, layerName_+"/weightsTensor")
	biases = variableManager.LoadOrCreateVariable(biasesValue, isTrainable_, layerName_+"/biasesTensor")
	return weights, biases


def CreateFcVariables(numberOfInputs_, numberOfOutputs_, isTrainable_, layerName_):
	weightsValue = tf.truncated_normal([numberOfInputs_, numberOfOutputs_],
					   mean=layerSettings.FC_WEIGHTS_RANDOM_MEAN,
					   stddev=layerSettings.FC_WEIGHTS_RANDOM_DEVIATION,
					   name=layerName_+"/createWeightsValues")
	biasesValue = tf.truncated_normal([numberOfOutputs_],
					  mean=layerSettings.FC_BIASES_RANDOM_MEAN,
					  stddev=layerSettings.FC_BIASES_RANDOM_DEVIATION,
					  name=layerName_+"/createBiasesValues")
	weights = variableManager.LoadOrCreateVariable(weightsValue, isTrainable_, layerName_ + "/weightsTensor")
	biases = variableManager.LoadOrCreateVariable(biasesValue, isTrainable_, layerName_ + "/biasesTensor")
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


