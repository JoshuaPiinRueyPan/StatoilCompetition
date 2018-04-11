import tensorflow as tf
from src.net.SubnetBase import SubnetBase
from src.layers.BasicLayers import *
import settings.OutputSettings as outSettings

VGG_INPUT_SIZE = [224, 224]
DARKNET19_MODEL_PATH = 'data/darknet19/darknet19.pb'

class DarkNet19(SubnetBase):
	def __init__(self, isTraining_, trainingStep_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.trainingStep = trainingStep_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.dropoutValue = 0.5

	def Build(self):
		darknet19_GraphDef = tf.GraphDef()

		darknetInput = self.transformInputToVGG_Input()

		with tf.name_scope("DarkNet19"):
			with open(DARKNET19_MODEL_PATH, 'rb') as modelFile:
				darknet19_GraphDef.ParseFromString(modelFile.read())
				listOfOperations = tf.import_graph_def(darknet19_GraphDef,
								    input_map={"input": darknetInput},
#								    return_elements=["BiasAdd_13"])
								    return_elements=["32-leaky"])
#								    return_elements=["BiasAdd_14"])
#								    return_elements=["34-leaky"])
#								    return_elements=["BiasAdd_15"])
#								    return_elements=["36-leaky"])
#								    return_elements=["BiasAdd_16"])
#								    return_elements=["38-leaky"])
#								    return_elements=["BiasAdd_17"])
#								    return_elements=["40-leaky"])
#								    return_elements=["Pad_18"])
#								    return_elements=["41-convolutional"])
#								    return_elements=["BiasAdd_18"])
				lastOp = listOfOperations[-1]
				print("\t lastOp.outputs = " + str(lastOp.outputs) )
				darknetOutput = lastOp.outputs[0]
			

		with tf.name_scope("Classifiyer"):
			net = FullyConnectedLayer('Fc1', darknetOutput, numberOfOutputs_=128)
			net, updateVariablesOp1 = BatchNormalization('BN1', net, isConvLayer_=False,
								     isTraining_=self.isTraining, currentStep_=self.trainingStep)
			net = LeakyRELU('LeakyRELU_2', net)

			net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

			net = FullyConnectedLayer('Fc2', net, numberOfOutputs_=128)
			net, updateVariablesOp2 = BatchNormalization('BN2', net, isConvLayer_=False,
								     isTraining_=self.isTraining, currentStep_=self.trainingStep)
			net = LeakyRELU('LeakyRELU_2', net)

			net = tf.cond(self.isTraining, lambda: tf.nn.dropout(net, self.dropoutValue), lambda: net)

			logits = FullyConnectedLayer('Fc3', net, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)

			updateVariablesOperations = tf.group(updateVariablesOp1, updateVariablesOp2)
			return logits, updateVariablesOperations


	def transformInputToVGG_Input(self):
		with tf.name_scope("InputProcessing"):
			imagesHH = self.inputImage[:, :, :, 0]
			imagesHV = self.inputImage[:, :, :, 1]
			imagesAngle = tf.zeros_like( imagesHH )
			totalImages = tf.stack( [imagesHH, imagesHV, imagesAngle] )
			totalImages = tf.transpose(totalImages, [1, 2, 3, 0])
			totalImages = tf.image.resize_images(totalImages, VGG_INPUT_SIZE)
			
			return  totalImages
	
