#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
import src.RadarImage
from src.DataManager import TrainingDataManager
from src.IceNet import *
import settings.TrainingSettings as trainSettings
import src.layers.LayerHelper as LayerHelper

class Solver:
	def __init__(self):
		# Data Manager
		self.dataManager = TrainingDataManager(trainSettings.TRAINING_SET_PATH_NAME, trainSettings.VALIDATION_RATIO)
		self.validation_x, self.validation_x_angle, self.validation_y = self.dataManager.GetValidationSet()
		self.NUMBER_OF_VALIDATION_DATA = self.validation_x.shape[0]

		# Net
		self.net = IceNet()
		self.updateNetOp = self.net.Build()
		self.trainLossOp, self.trainAccuOp = self.net.GetTrainingResultTensor()
		self.validaLossOp, self.validaAccuOp = self.net.GetValidationResultTensor()

		# Optimizer
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		try:
			# If there's other losses (e.g. Regularization Loss)
			otherLossOp = tf.losses.get_total_loss(add_regularization_losses=True)
			totalLossOp = self.trainLossOp + otherLossOp
		except:
			# If there's no other loss op
			totalLossOp = self.trainLossOp
		self.optimzeOp = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(totalLossOp)

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
		summary, lossValue, accuValue  =  session.run( [self.summaryOp, self.trainLossOp, self.trainAccuOp],
								feed_dict={	self.net.isTraining : False,
										self.net.trainingStep : self.dataManager.step,
										self.net.inputImage : batch_x,
										self.net.inputAngle : batch_x_angle,
										self.net.groundTruth : batch_y})

		self.trainSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    train:")
		print("        loss: " + str(lossValue) + ", accuracy: " + str(accuValue) + "\n")


	def CalculateValidation(self, session, shouldSaveSummary):
		summary, lossValue, accuValue  =  session.run( [self.summaryOp, self.validaLossOp, self.validaAccuOp],
								feed_dict={	self.net.isTraining : False,
										self.net.trainingStep : self.dataManager.step,
										self.net.inputImage : self.validation_x,
										self.net.inputAngle : self.validation_x_angle,
										self.net.groundTruth : self.validation_y})
		if shouldSaveSummary:
			self.validaSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    validation:")
		print("        loss: " + str(lossValue) + ", accuracy: " + str(accuValue) + "\n")

	def saveCheckpoint(self, tf_session):
		pathToSaveCheckpoint = os.path.join(trainSettings.PATH_TO_SAVE_MODEL, "save_epoch_" + str(self.dataManager.epoch) )
		checkpointPathFileName = os.path.join(pathToSaveCheckpoint, "IceNet.ckpt")
		self.saver.save(tf_session, checkpointPathFileName)



if __name__ == "__main__":
	solver = Solver()
	solver.Run()
