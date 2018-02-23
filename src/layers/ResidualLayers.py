from src.layers.BasicLayers import *


def ResidualBlock(layerName_, inputTensor_, listOfConvFilterSize_, isTraining_, trainingStep_, activationType_="RELU"):
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
		# Block 'Conv1x1a'
		blockName = layerName_ + "/Conv1x1a"
		residualOp = ConvLayer(	blockName+'/Conv_1x1', inputTensor_,
					filterSize_=1, numberOfFilters_=listOfConvFilterSize_[0])
		residualOp, updateVariablesOp1 = BatchNormalization(blockName+"/BN_1x1", residualOp, isConvLayer_=True,
								    isTraining_=isTraining_, currentStep_=trainingStep_)
		residualOp = SetActivation(blockName+"/"+activationType_, residualOp, activationType_)

		# Block 'Conv3x3b'
		blockName = layerName_ + "/Conv3x3b"
		residualOp = ConvLayer(	blockName+'/Conv3x3', residualOp,
					filterSize_=3, numberOfFilters_=listOfConvFilterSize_[1])
		residualOp, updateVariablesOp2 = BatchNormalization(blockName+"/BN_3x3", residualOp, isConvLayer_=True,
								    isTraining_=isTraining_, currentStep_=trainingStep_)
		residualOp = SetActivation(blockName+"/"+activationType_, residualOp, activationType_)

		# Block 'Conv1x1c'
		blockName = layerName_ + "/Conv1x1c"
		residualOp = ConvLayer(	blockName+'/Conv1x1', residualOp,
					filterSize_=1, numberOfFilters_=listOfConvFilterSize_[2])
		residualOp, updateVariablesOp3 = BatchNormalization(blockName+"/BN_1x1", residualOp, isConvLayer_=True, 
								    isTraining_=isTraining_, currentStep_=trainingStep_)
		residualOp = SetActivation(blockName+"/"+activationType_, residualOp, activationType_)

		output = tf.add(inputTensor_, residualOp, name=layerName_+"/add")
		updateOperations = tf.group( updateVariablesOp1, updateVariablesOp2, updateVariablesOp3)

		return output, updateOperations

	else:
		errorMessage = "The input parameter 'listOfConvFilterSize_' of ResidualBlock() shuold only have THREE elements,\n"
		errorMessage += "However, your input = '" + str(listOfConvFilterSize_) + "'\n"
		errorMessage += "You check src/layers/ResidualLayers.py for more information"
		raise ValueError(errorMessage)


def ResidualHeadBlock(layerName_, inputTensor_, listOfConvFilterSize_, isTraining_, trainingStep_, activationType_="RELU"):
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
		# Block 'Conv1x1a'
		blockName = layerName_ + "/Conv1x1a"
		residualOp = ConvLayer(	blockName+'/Conv1x1', inputTensor_,
					 filterSize_=1,	numberOfFilters_=listOfConvFilterSize_[0])
		residualOp, updateVariablesOp1 = BatchNormalization(blockName+"/BN_1x1", residualOp, isConvLayer_=True,
								    isTraining_=isTraining_, currentStep_=trainingStep_)
		residualOp = SetActivation(blockName+"/"+activationType_, residualOp, activationType_)

		# Block 'Conv3x3b'
		blockName = layerName_ + '/Conv3x3b'
		residualOp = ConvLayer(	blockName+'/Conv3x3', residualOp,
					filterSize_=3,	numberOfFilters_=listOfConvFilterSize_[1])
		residualOp, updateVariablesOp2 = BatchNormalization(blockName+"/BN_3x3", residualOp, isConvLayer_=True,
								    isTraining_=isTraining_, currentStep_=trainingStep_)
		residualOp = SetActivation(blockName+"/"+activationType_, residualOp, activationType_)

		# Block 'Conv1x1c'
		blockName = layerName_ + "/Conv1x1c"
		residualOp = ConvLayer( blockName+'/Conv1x1', residualOp,
					filterSize_=1, numberOfFilters_=listOfConvFilterSize_[2])
		residualOp, updateVariablesOp3 = BatchNormalization(blockName+"/BN_1x1", residualOp, isConvLayer_=True,
								    isTraining_=isTraining_, currentStep_=trainingStep_)
		residualOp = SetActivation(blockName+"/"+activationType_, residualOp, activationType_)

		# Block 'Conv1x1_Identity'
		blockName = layerName_ + '/Conv1x1_Identity'
		identityOp = ConvLayer(	blockName+'/Conv1x1_I', inputTensor_,
					filterSize_=1, numberOfFilters_=listOfConvFilterSize_[2])
		identityOp, updateVariablesOp4 = BatchNormalization(blockName+"/BN_I", identityOp, isConvLayer_=True,
								    isTraining_=isTraining_, currentStep_=trainingStep_)
		identityOp = SetActivation(blockName+"/"+activationType_, identityOp, activationType_)

		output = tf.add(identityOp, residualOp, name=layerName_+"/add")
		updateOperations = tf.group( updateVariablesOp1, updateVariablesOp2, updateVariablesOp3, updateVariablesOp4)

		return output, updateOperations

	else:
		errorMessage = "The input parameter 'listOfConvFilterSize_' of ResidualHeadBlock()"
		errorMessage += " shuold only have THREE elements,\n"
		errorMessage += "You check src/layers/ResidualLayers.py for more information"
		raise ValueError(errorMessage)



def ResidualLayer(layerName_, inputTensor_, numberOfResidualBlocks_, listOfConvFilterSize_,
		  isTraining_, trainingStep_, activationType_="RELU"):
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
		headOutput, updateHead = ResidualHeadBlock( layerName_+"/HeadBlock",
							    inputTensor_, listOfConvFilterSize_, 
							    isTraining_, trainingStep_, activationType_)
		listOfUpdateOperations.append(updateHead)
		preiousOutput = headOutput
		for i in range(numberOfResidualBlocks_ - 1):  # The first Block is created above, thus '-1'
			preiousOutput, currentUpdate = ResidualBlock(layerName_+"/BodyBlock"+str(i),
								     preiousOutput, listOfConvFilterSize_,
								     isTraining_, trainingStep_, activationType_,
								    )
			listOfUpdateOperations.append(currentUpdate)
		
		updateOperations = tf.group(*listOfUpdateOperations)

		return preiousOutput, updateOperations

	else:
		errorMessage = "The input parameter 'numberOfResidualBlocks_' of ResidualLayer shold be LARGE than ONE.\n"
		errorMessage += "You check src/layers/ResidualLayers.py for more information"
		raise ValueError(errorMessage)
