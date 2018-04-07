#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
import src.RadarImage
from src.DataManager import TrainingDataManager
from src.IceNet import *
import settings.TrainSettings as trainSettings
import src.layers.LayerHelper as LayerHelper

class Solver:
	def __init__(self):
		# Data Manager
		self.dataManager = TrainingDataManager(trainSettings.TRAINING_SET_PATH_NAME, trainSettings.VALIDATION_RATIO)
		self.validation_x, self.validation_x_angle, self.validation_y = self.dataManager.GetValidationSet()

		# Net
		self.net = IceNet()
		self.lossOp, self.accuracyOp, self.updateNetOp = self.net.Build()

		# Optimizer
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		try:
			# If there's other losses (e.g. Regularization Loss)
			otherLossOp = tf.losses.get_total_loss(add_regularization_losses=True)
			totalLossOp = self.lossOp + otherLossOp
		except:
			# If there's no other loss op
			totalLossOp = self.lossOp

		#self.optimzeOp = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(totalLossOp)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
		gradients = optimizer.compute_gradients(totalLossOp)
		self.drawGradients(gradients)
		self.optimzeOp = optimizer.apply_gradients(gradients)

		# Summary
		self.trainSumWriter = tf.summary.FileWriter(trainSettings.PATH_TO_SAVE_MODEL+"/train")
		self.validaSumWriter = tf.summary.FileWriter(trainSettings.PATH_TO_SAVE_MODEL+"/valid")
		self.saver = tf.train.Saver(max_to_keep=trainSettings.MAX_TRAINING_SAVE_MODEL)
		self.summaryOp = tf.summary.merge_all()

	def Run(self):
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)
			self.validaSumWriter.add_graph(sess.graph)
			self.recoverFromPretrainModelIfRequired(sess)

			# Calculate Validation before Training
			print("Validation before Training  ======================================")
			self.CalculateValidation(sess, shouldSaveSummary=False)

			while self.dataManager.epoch < trainSettings.MAX_TRAINING_EPOCH:
				batch_x, batch_x_angle, batch_y = self.dataManager.GetTrainingBatch(trainSettings.BATCH_SIZE)
				self.trainIceNet(sess, batch_x, batch_x_angle, batch_y)
				self.updateIceNet(sess, batch_x, batch_x_angle, batch_y)

				if self.dataManager.isNewEpoch:
					print("Epoch: " + str(self.dataManager.epoch)+" ======================================")
					self.CalculateTrainingLoss(sess, batch_x, batch_x_angle, batch_y)
					self.CalculateValidation(sess, shouldSaveSummary=True)

					if self.dataManager.epoch >= trainSettings.EPOCHS_TO_START_SAVE_MODEL:
						self.saveCheckpoint(sess)
			print("Optimization finished!")


	def recoverFromPretrainModelIfRequired(self, session):
		if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
			print("Load Pretrain model from: " + trainSettings.PRETRAIN_MODEL_PATH_NAME)
			listOfAllVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			variablesToBeRecovered = [ eachVariable for eachVariable in listOfAllVariables \
						   if eachVariable.name.split('/')[0] not in \
						   trainSettings.NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT ]
			modelLoader = tf.train.Saver(variablesToBeRecovered)
			modelLoader.restore(session, trainSettings.PRETRAIN_MODEL_PATH_NAME)


	def trainIceNet(self, session, batch_x, batch_x_angle, batch_y):
		currentLearningRate = trainSettings.GetLearningRate(self.dataManager.epoch)
		session.run( self.optimzeOp,
			     feed_dict={self.net.isTraining : True,
					self.net.trainingStep : self.dataManager.step,
					self.net.inputImage : batch_x,
					self.net.inputAngle : batch_x_angle,
					self.net.groundTruth : batch_y,
					self.learningRate : currentLearningRate})

	def updateIceNet(self, session, batch_x, batch_x_angle, batch_y):
		'''
		    Some Network has variables that need to be updated after training (e.g. the net with
		    batch normalization).  After training, following code update such variables.
		'''
		session.run( self.updateNetOp,
			     feed_dict={self.net.isTraining : False,
					self.net.trainingStep : self.dataManager.step,
					self.net.inputImage : batch_x,
					self.net.inputAngle : batch_x_angle,
					self.net.groundTruth : batch_y})


	def CalculateTrainingLoss(self, session, batch_x, batch_x_angle, batch_y):
		summaryValue, lossValue, accuValue  =  session.run( [self.summaryOp, self.lossOp, self.accuracyOp],
								    feed_dict={	self.net.isTraining : False,
										self.net.trainingStep : self.dataManager.step,
										self.net.inputImage : batch_x,
										self.net.inputAngle : batch_x_angle,
										self.net.groundTruth : batch_y})

		summary = tf.Summary()
		summary.ParseFromString(summaryValue)
		summary.value.add(tag='loss', simple_value=lossValue)
		summary.value.add(tag='accuracy', simple_value=accuValue)

		self.trainSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    train:")
		print("        loss: " + str(lossValue) + ", accuracy: " + str(accuValue) + "\n")


	def CalculateValidation(self, session, shouldSaveSummary):
		if trainSettings.NUMBER_OF_VALIDATION_DATA <= trainSettings.BATCH_SIZE:
			self.calculateValidationBywholeBatch(session, shouldSaveSummary)

		else:
			self.calculateValidationOneByOne(session, shouldSaveSummary)

	def calculateValidationBywholeBatch(self, session, shouldSaveSummary):
		summaryValue, lossValue, accuValue  =  session.run( [self.summaryOp, self.lossOp, self.accuracyOp],
								    feed_dict={	self.net.isTraining : False,
										self.net.trainingStep : self.dataManager.step,
										self.net.inputImage : self.validation_x,
										self.net.inputAngle : self.validation_x_angle,
										self.net.groundTruth : self.validation_y})
		if shouldSaveSummary:
			summary = tf.Summary()
			summary.ParseFromString(summaryValue)
			summary.value.add(tag='loss', simple_value=lossValue)
			summary.value.add(tag='accuracy', simple_value=accuValue)
			self.validaSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    validation:")
		print("        loss: " + str(lossValue) + ", accuracy: " + str(accuValue) + "\n")

	def calculateValidationOneByOne(self, session, shouldSaveSummary):
		'''
		When deal with a Large Model, stuff all validation set into a batch is not possible.
		Therefore, following stuff each validation data at a time
		'''
		arrayOfValidaLoss = np.zeros( (trainSettings.NUMBER_OF_VALIDATION_DATA) )
		arrayOfValidaAccu = np.zeros( (trainSettings.NUMBER_OF_VALIDATION_DATA) )
		for i in range(trainSettings.NUMBER_OF_VALIDATION_DATA):
			validaImage = self.validation_x[i]
			validaImage = np.reshape(validaImage,
						 [1, RadarImage.DATA_WIDTH, RadarImage.DATA_HEIGHT, RadarImage.DATA_CHANNELS])

			validaAngle = self.validation_x_angle[i]
			validaAngle = np.reshape(validaAngle, [1, 1])

			validaLabel = self.validation_y[i]
			validaLabel = np.reshape(validaLabel, [1, 2])
			lossValue, accuValue  =  session.run( [ self.lossOp, self.accuracyOp],
							      feed_dict={	self.net.isTraining : False,
										self.net.trainingStep : self.dataManager.step,
										self.net.inputImage : validaImage,
										self.net.inputAngle : validaAngle,
										self.net.groundTruth : validaLabel})
			arrayOfValidaLoss[i] = lossValue
			arrayOfValidaAccu[i] = accuValue

		meanLoss = np.mean(arrayOfValidaLoss)
		meanAccu = np.mean(arrayOfValidaAccu)

		if shouldSaveSummary:
			summary = tf.Summary()
			summary.value.add(tag='loss', simple_value=meanLoss)
			summary.value.add(tag='accuracy', simple_value=meanAccu)
			self.validaSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    validation:")
		print("        loss: " + str(meanLoss) + ", accuracy: " + str(meanAccu) + "\n")


	def saveCheckpoint(self, tf_session):
		pathToSaveCheckpoint = os.path.join(trainSettings.PATH_TO_SAVE_MODEL, "save_epoch_" + str(self.dataManager.epoch) )
		checkpointPathFileName = os.path.join(pathToSaveCheckpoint, "IceNet.ckpt")
		self.saver.save(tf_session, checkpointPathFileName)

	def drawGradients(self, gradientsInfo_):
		for eachGradient, eachVariable in gradientsInfo_:
			if eachGradient is not None:
				tf.summary.histogram(eachVariable.op.name + '/gradient', eachGradient)


if __name__ == "__main__":
	solver = Solver()
	solver.Run()
