#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
from src.DataManager import TrainingDataManager
from src.IceNet import *
import settings.TrainingSettings as trainSettings
from src.OptimizerFactory import *

class Solver:
	def __init__(self):
		# Data Manager
		self.dataManager = TrainingDataManager(trainSettings.TRAINING_SET_PATH_NAME, trainSettings.VALIDATION_RATIO)
		self.validation_x, self.validation_x_angle, self.validation_y = self.dataManager.GetValidationSet()

		# Net
		self.net = IceNet()
		self.cost, self.accuracy = self.net.Build()

		# Optimizer
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)

		# Saver
		self.saver = tf.train.Saver()

	def Run(self):
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)

			while self.dataManager.epoch < trainSettings.MAX_TRAINING_EPOCH:
				batch_x, batch_x_angle, batch_y = self.dataManager.GetTrainingBatch(trainSettings.BATCH_SIZE)
				print("angle = " + str(batch_x_angle))
				self.trainIcenet(sess, batch_x, batch_x_angle, batch_y)

				if self.dataManager.isNewEpoch:
					print("Epoch: " + str(self.dataManager.epoch)+" ======================================")
					self.calculateTrainingLoss(sess, batch_x, batch_x_angle, batch_y)
					self.calculateValidationLoss(sess, batch_x, batch_x_angle, batch_y)

					if self.dataManager.epoch >= trainSettings.EPOCHS_TO_START_SAVE_WEIGHTINGS:
						pathToSaveCheckpoint = os.path.join("temp", 
										     "save_epoch",
										     "iceberg.ckpt")
						self.saver.save(sess,  pathToSaveCheckpoint, global_step=self.dataManager.epoch)
			print("Optimization finished!")

	def trainIcenet(self, session, batch_x, batch_x_angle, batch_y):
		currentLearningRate = trainSettings.GetLearningRate(self.dataManager.epoch)
		session.run(self.optimizer, feed_dict={self.net.isTraining : True,
							self.net.inputImage : batch_x,
							self.net.inputAngle : batch_x_angle,
							self.net.groundTruth : batch_y,
							self.learningRate : currentLearningRate})


	def calculateTrainingLoss(self, session, batch_x, batch_x_angle, batch_y):
		loss, accuracy = session.run( [self.cost, self.accuracy],
					   feed_dict={self.net.isTraining : False,
						      self.net.inputImage : batch_x,
						      self.net.inputAngle : batch_x_angle,
						      self.net.groundTruth : batch_y})
		print("    train:")
		print("        loss: " + str(loss) + ", accuracy: " + str(accuracy)+"\n")

	def calculateValidationLoss(self, session, batch_x, batch_x_angle, batch_y):
		loss, accuracy = session.run( [self.cost, self.accuracy],
					   feed_dict={self.net.isTraining : False,
						      self.net.inputImage : self.validation_x,
						      self.net.inputAngle : self.validation_x_angle,
						      self.net.groundTruth : self.validation_y})
		print("    validation:")
		print("        loss: " + str(loss) + ", accuracy: " + str(accuracy))



if __name__ == "__main__":
	solver = Solver()
	solver.Run()
