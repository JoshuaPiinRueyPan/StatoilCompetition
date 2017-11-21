#!/usr/bin/python
import tensorflow as tf
import numpy as np
from src.DataManager import *
from src.IceNet import *

learning_rate = 0.0001
BATCH_SIZE = 100


TRAINING_SET_PATH_NAME = "data/train.json"
VALIDATION_RATIO = 0.1
EPOCHS_TO_START_SAVE_WEIGHTINGS = 100
MAX_TRAINING_EPOCH = 200

class Solver:
	def __init__(self):
		self.net = IceNet()
		self.cost, self.accuracy = self.net.Build()
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
		self.dataManager = DataManager(TRAINING_SET_PATH_NAME, VALIDATION_RATIO)
		self.saver = tf.train.Saver()
		self.validation_x, self.validation_x_angle, self.validation_y = self.dataManager.GetValidationSet()

	def Run(self):
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)

			while self.dataManager.epoch < MAX_TRAINING_EPOCH:
				batch_x, batch_x_angle, batch_y = self.dataManager.GetTrainingBatch(BATCH_SIZE)

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
					if self.dataManager.epoch >= EPOCHS_TO_START_SAVE_WEIGHTINGS:
						self.saver.save(sess, "save_epoch" + str(self.dataManager.epoch) + "/iceberg.ckpt")
			print("Optimization finished!")


if __name__ == "__main__":
	solver = Solver()
	solver.Run()
