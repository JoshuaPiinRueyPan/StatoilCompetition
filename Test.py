#!/usr/bin/python
import os
import tensorflow as tf
import numpy as np
from src.DataManager import TestingDataManager
from src.IceNet import *
import settings.TestSettings as testSettings
import csv

class Testing:
	def __init__(self):
		self.net = IceNet()
		self.cost, self.accuracy, _ = self.net.Build()
		print("Loading Test Set, please wait...")
		self.dataManager = TestingDataManager(testSettings.TESTING_SET_PATH_NAME)
		print("Loading finished.")
		self.saver = tf.train.Saver()
		self.listOfIDs, self.arrayOfImages, self.arrayOfAngles = self.dataManager.GetTestingSet()

	def Run(self):
		outputFile = open(testSettings.SUMMARY_FILE_PATH_NAME, 'w')
		csvWriter = csv.writer(outputFile)
		csvHeader = ['id', 'is_iceberg']
		csvWriter.writerow(csvHeader)
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)
			self.saver.restore(sess, testSettings.MODEL_PATH_NAME)
			for i in range(len(self.listOfIDs)):
				netOutput = sess.run(self.net.softmax, feed_dict={self.net.isTraining : False,
								    self.net.inputImage : [self.arrayOfImages[i]],
								    self.net.inputAngle : [self.arrayOfAngles[i]]})
				answer = testSettings.GetAnswer(netOutput)
				answerList = [self.listOfIDs[i], answer]
				csvWriter.writerow(answerList)

			print("Finished!")


if __name__ == "__main__":
	testing = Testing()
	testing.Run()
