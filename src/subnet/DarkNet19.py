import tensorflow as tf
from src.subnet.SubnetBase import SubnetBase
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

	def Build(self):
		darknet19_GraphDef = tf.GraphDef()

		darknetInput = self.transformInputToVGG_Input()

		with tf.name_scope("DarkNet19"):
			with open(DARKNET19_MODEL_PATH, 'rb') as modelFile:
				darknet19_GraphDef.ParseFromString(modelFile.read())
				listOfOperations = tf.import_graph_def(darknet19_GraphDef,
								    input_map={"input": darknetInput},
								    return_elements=["41-convolutional"])
				lastOp = listOfOperations[-1]
				print("\t lastOp.outputs = " + str(lastOp.outputs) )
				darknetOutput = lastOp.outputs[0]
			

		logits = FullyConnectedLayer('Fc-Final', darknetOutput, numberOfOutputs_=outSettings.NUMBER_OF_CATEGORIES)

		print("All Trainable Variables:\n" + str( tf.trainable_variables() ) )

		return logits, tf.no_op()

	def transformInputToVGG_Input(self):
		with tf.name_scope("InputProcessing"):
			imagesHH = self.inputImage[:, :, :, 0]
			imagesHV = self.inputImage[:, :, :, 1]
			imagesAngle = tf.zeros_like( imagesHH )
			totalImages = tf.stack( [imagesHH, imagesHV, imagesAngle] )
			totalImages = tf.transpose(totalImages, [1, 2, 3, 0])
			totalImages = tf.image.resize_images(totalImages, VGG_INPUT_SIZE)
			
			return  totalImages
	
