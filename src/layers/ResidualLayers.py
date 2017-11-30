from src.layers.BasicLayers import *


def ResidualBlock(isTraining_, trainingStep_, inputTensor_, listOfConvFilterSize_, activationType_="RELU", layerName_="ResBlock"):
	with tf.variable_scope(layerName_):
		'''
		    This function will create Residual Block as follows:
		    (supposed that listOfConvFilterSize_ = [a, b, c])
					|			
				----------------|		
				|		|		
				|	  Conv 1x1, a		
				|	  Conv 3x3, b		
				|	  Conv 1x1, c		
				|      ---	|		
				|------|+|------|		
				       ---			
					|			

		    For example, to build the block in the second layer in ResNet:
			 listOfConvFilterSize_ = [64, 64, 256]
		'''
		if len(listOfConvFilterSize_) == 3:
			residualOp = ConvLayer(inputTensor_, filterSize_=1, numberOfFilters_=listOfConvFilterSize_[0], layerName_='/Conv1x1')
			residualOp, updateVariablesOp1 = BatchNormalization(isTraining_, trainingStep_, residualOp, isConvLayer_=True)
			residualOp = SetActivation(residualOp, activationType_)

			residualOp = ConvLayer(residualOp, filterSize_=3, numberOfFilters_=listOfConvFilterSize_[1], layerName_='/Conv3x3')
			residualOp, updateVariablesOp2 = BatchNormalization(isTraining_, trainingStep_, residualOp, isConvLayer_=True)
			residualOp = SetActivation(residualOp, activationType_)

			residualOp = ConvLayer(residualOp, filterSize_=1, numberOfFilters_=listOfConvFilterSize_[2], layerName_='/Conv1x1')
			residualOp, updateVariablesOp3 = BatchNormalization(isTraining_, trainingStep_, residualOp, isConvLayer_=True)
			residualOp = SetActivation(residualOp, activationType_)

			output = inputTensor_ + residualOp
			updateOperations = tf.group( updateVariablesOp1, updateVariablesOp2, updateVariablesOp3)

			return output, updateOperations

		else:
			errorMessage = "The input parameter 'listOfConvFilterSize_' of ResidualBlock() shuold only have THREE elements,\n"
			errorMessage += "However, your input = '" + str(listOfConvFilterSize_) + "'\n"
			errorMessage += "You check src/layers/ResidualLayers.py for more information"
			raise ValueError(errorMessage)

def _residualHeadBlock(isTraining_, trainingStep_, inputTensor_, listOfConvFilterSize_, activationType_="RELU", layerName_="ResBlock"):
	with tf.variable_scope(layerName_):
		'''
		    This function will create Residual Head Block as follows:
		    (supposed that listOfConvFilterSize_ = [a, b, c])
					|			\
				----------------|		|
				|		|		|
				|	  Conv 1x1, a		|
			  Conv 1x1 c	  Conv 3x3, b		|  The Head Block is slightly different
				|	  Conv 1x1, c		|  than the Boddy Block (has one more Conv
				|		|		|  on the Left).
				|      ---	|		|
				|------|+|------|		/
				       ---			
					|			
		    Note: Directly call the ResidualLayer() if you want to build the ResidualLayer 
		'''
		if len(listOfConvFilterSize_) == 3:
			residualOp = ConvLayer(inputTensor_, filterSize_=1, numberOfFilters_=listOfConvFilterSize_[0], layerName_='/Conv1x1')
			residualOp, updateVariablesOp1 = BatchNormalization(isTraining_, trainingStep_, residualOp, isConvLayer_=True)
			residualOp = SetActivation(residualOp, activationType_)

			residualOp = ConvLayer(residualOp, filterSize_=3, numberOfFilters_=listOfConvFilterSize_[1], layerName_='/Conv3x3')
			residualOp, updateVariablesOp2 = BatchNormalization(isTraining_, trainingStep_, residualOp, isConvLayer_=True)
			residualOp = SetActivation(residualOp, activationType_)

			residualOp = ConvLayer(residualOp, filterSize_=1, numberOfFilters_=listOfConvFilterSize_[2], layerName_='/Conv1x1')
			residualOp, updateVariablesOp3 = BatchNormalization(isTraining_, trainingStep_, residualOp, isConvLayer_=True)
			residualOp = SetActivation(residualOp, activationType_)

			identityOp = ConvLayer(inputTensor_, filterSize_=1, numberOfFilters_=listOfConvFilterSize_[2], layerName_='/Conv1x1')
			identityOp, updateVariablesOp4 = BatchNormalization(isTraining_, trainingStep_, identityOp, isConvLayer_=True)
			identityOp = SetActivation(identityOp, activationType_)

			output = identityOp + residualOp
			updateOperations = tf.group( updateVariablesOp1, updateVariablesOp2, updateVariablesOp3, updateVariablesOp4)

			return output, updateOperations

		else:
			errorMessage = "The input parameter 'listOfConvFilterSize_' of _residualHeadBlock()"
			errorMessage += " shuold only have THREE elements,\n"
			errorMessage += "You check src/layers/ResidualLayers.py for more information"
			raise ValueError(errorMessage)



def ResidualLayer(isTraining_, trainingStep_, inputTensor_, numberOfResidualBlocks_, listOfConvFilterSize_,
		  activationType_="RELU", layerName_="ResLayer"):
	with tf.variable_scope(layerName_):
		'''
		    This function is the wrapper that use the above block to build the Layers in ResNet.
		    The ResLayer has following configuration:
		    (supposed that: listOfConvFilterSize_=[ a, b, c ] )
			
					|			\
				----------------|		|
				|		|		|
				|	  Conv 1x1, a		|
			  Conv 1x1 c	  Conv 3x3, b		|  The Head Block is slightly different
				|	  Conv 1x1, c		|  than the Boddy Block (has one more Conv
				|		|		|  on the Left).
				|      ---	|		|
				|------|+|------|		/
				       ---			
					|			
				----------------|		\
				|		|		|
				|	  Conv 1x1, a		|
				|	  Conv 3x3, b		|  The Body Block may repeat multiple times
				|	  Conv 1x1, c		|  (numberOfResidualBlocks_ - 1)
				|		|		|
				|      ---	|		|
				|------|+|------|		/
				       ---			
					|
		'''
		if numberOfResidualBlocks_ > 1:
			listOfUpdateOperations = []
			headOutput, updateHead = _residualHeadBlock(isTraining_, trainingStep_, inputTensor_,
								    listOfConvFilterSize_, activationType_,
								    layerName_+"/HeadBlock")
			listOfUpdateOperations.append(updateHead)
			currentInput = headOutput
			for i in range(numberOfResidualBlocks_):
				currentInput, currentUpdate = ResidualBlock(isTraining_, trainingStep_, currentInput,
									    listOfConvFilterSize_, activationType_,
									    layerName_+"/BodyBlock")
				listOfUpdateOperations.append(currentUpdate)
			
			output = headOutput + currentInput
			updateOperations = tf.group(*listOfUpdateOperations)

			return output, updateOperations

		else:
			errorMessage = "The input parameter 'numberOfResidualBlocks_' of ResidualLayer shold be LARGE than ONE.\n"
			errorMessage += "You check src/layers/ResidualLayers.py for more information"
			raise ValueError(errorMessage)
