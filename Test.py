#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
from src.DataManager import *
from src.IceNet import *
import settings.TestSettings as testSettings
import csv

class Testing:
	def __init__(self):
		self.net = IceNet()
		self.cost, self.accuracy = self.net.Build()
		#self.optimizer = tf.train.AdamOptimizer(learning_rate=trainSettings.learningRate).minimize(self.cost)
		self.dataManager = DataManager(testSettings.TESTING_SET_PATH_NAME, 0)
		self.saver = tf.train.Saver()
		#self.validation_x, self.validation_x_angle, self.validation_y = self.dataManager.GetValidationSet()
		self.listOfIDs, self.arrayOfImages, self.arrayOfAngles = self.dataManager.GetTestingSet()
	def Run(self):
		csvfile = open('temp/answerlog.csv', 'w')
		csvWriter = csv.writer(csvfile)
		csvHeader = ['id', 'is_iceberg']
		csvWriter.writerow(csvHeader)
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)
			self.saver.restore(sess, "temp/save_epoch300/iceberg.ckpt")
			for i in range(len(self.listOfIDs)):
				answer = sess.run(self.net.softmax, feed_dict={self.net.isTraining : False,
								    self.net.inputImage : [self.arrayOfImages[i]],
								    self.net.inputAngle : [self.arrayOfAngles[i]]})
				answerList = [self.listOfIDs[i], answer[0][1]]
				csvWriter.writerow(answerList)

			print("Finished!")


if __name__ == "__main__":
	testing = Testing()
	testing.Run()
