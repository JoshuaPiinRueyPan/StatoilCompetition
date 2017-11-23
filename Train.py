#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
from src.DataManager import TrainingDataManager
from src.IceNet import *
import settings.TrainingSettings as trainSettings

class Solver:
	def __init__(self):
		self.net = IceNet()
		self.cost, self.accuracy = self.net.Build()
		self.optimizer = tf.train.AdamOptimizer(learning_rate=trainSettings.learningRate).minimize(self.cost)
		self.dataManager = TrainingDataManager(trainSettings.TRAINING_SET_PATH_NAME, trainSettings.VALIDATION_RATIO)
		self.saver = tf.train.Saver()
		self.validation_x, self.validation_x_angle, self.validation_y = self.dataManager.GetValidationSet()

	def Run(self):
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)

			while self.dataManager.epoch < trainSettings.MAX_TRAINING_EPOCH:
				batch_x, batch_x_angle, batch_y = self.dataManager.GetTrainingBatch(trainSettings.BATCH_SIZE)

				sess.run(self.optimizer, feed_dict={self.net.isTraining : True,
								    self.net.inputImage : batch_x,
								    self.net.inputAngle : batch_x_angle,
								    self.net.groundTruth : batch_y})
				if self.dataManager.isNewEpoch:
					print("Epoch: " + str(self.dataManager.epoch)+" ======================================")
					loss, accuracy = sess.run( [self.cost, self.accuracy],
								   feed_dict={self.net.isTraining : False,
									      self.net.inputImage : batch_x,
									      self.net.inputAngle : batch_x_angle,
									      self.net.groundTruth : batch_y})
					print("    train:")
					print("        loss: " + str(loss) + ", accuracy: " + str(accuracy)+"\n")
					loss, accuracy = sess.run( [self.cost, self.accuracy],
								   feed_dict={self.net.isTraining : False,
									      self.net.inputImage : self.validation_x,
									      self.net.inputAngle : self.validation_x_angle,
									      self.net.groundTruth : self.validation_y})
					print("    validation:")
					print("        loss: " + str(loss) + ", accuracy: " + str(accuracy))
					if self.dataManager.epoch >= trainSettings.EPOCHS_TO_START_SAVE_WEIGHTINGS:
						pathToSaveCheckpoint = os.path.join("temp", 
										     "save_epoch" + str(self.dataManager.epoch),
										     "iceberg.ckpt")
						self.saver.save(sess,  pathToSaveCheckpoint)
			print("Optimization finished!")


if __name__ == "__main__":
	solver = Solver()
	solver.Run()
