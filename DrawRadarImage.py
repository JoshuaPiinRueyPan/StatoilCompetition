#!/usr/bin/python
import sys
import cv2
import os
from src.RadarImage import *

class DataDrawer:
	def Draw(self, listOfRadarImage_, TARGET_DIRECTORY_):

		self.createDirectory(TARGET_DIRECTORY_)
		for eachRadarImage in listOfRadarImage_:
			print("\tDraw: " + eachRadarImage.name)
			self.drawEachRadarImage(eachRadarImage, TARGET_DIRECTORY_)

	def __init__(self):
		self.SCALE_OF_TARGET_IMAGE = 8

	def createDirectory(self, directoryName_):
		if not os.path.exists(directoryName_):
			os.mkdir(directoryName_)

	def drawEachRadarImage(self, radarImage_, TARGET_DIRECTORY_):
		imageName = radarImage_.name

		imageHH = radarImage_.GetImageHH()
		imageHH = self.resizeImageToTargetSize(imageHH)
		self.appendInfoToImage("HH", imageHH, radarImage_)

		imageHV = radarImage_.GetImageHV()
		imageHV = self.resizeImageToTargetSize(imageHV)
		self.appendInfoToImage("HV", imageHV, radarImage_)

		imageTotal = radarImage_.GetTotalImage()
		imageTotal = self.resizeImageToTargetSize(imageTotal)
		self.appendInfoToImage("Total", imageTotal, radarImage_)

		self.saveEachRadarImage(TARGET_DIRECTORY_, radarImage_, imageHH, imageHV, imageTotal)
			
	def resizeImageToTargetSize(self, image_):
		return cv2.resize(image_, (self.SCALE_OF_TARGET_IMAGE*image_.shape[0], self.SCALE_OF_TARGET_IMAGE*image_.shape[1]) )

	def appendInfoToImage(self, prefix_, image_, radarImage_):
		categoryInfo = "?"
		if radarImage_.hasAnswer:
			if radarImage_.isIceberg:
				categoryInfo = "ice"
			else:
				categoryInfo = "ship"
		angleInfo = "NA"
		if radarImage_.hasAngle:
			angleInfo = "{0:.2f}".format(radarImage_.angle)
		imageInfo = prefix_  + ";  "  + categoryInfo + ";  angle: " + angleInfo
		cv2.putText(image_, imageInfo, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

	def saveEachRadarImage(self, TARGET_DIRECTORY_, radarImage_, imageHH_, imageHV_, imageTotal_):
		pathToSaveImageHH = radarImage_.name + "_HH.jpg"
		pathToSaveImageHV = radarImage_.name + "_HV.jpg"
		pathToSaveTotalImage = radarImage_.name + "_total.jpg"
		if not radarImage_.hasAngle:
			newDirectoryName = "NoAngle"
			newDirectoryPath = os.path.join(TARGET_DIRECTORY_, newDirectoryName)
			self.createDirectory(newDirectoryPath)
			pathToSaveImageHH = os.path.join(newDirectoryPath, pathToSaveImageHH )
			pathToSaveImageHV = os.path.join(newDirectoryPath, pathToSaveImageHV )
			pathToSaveTotalImage = os.path.join(newDirectoryPath, pathToSaveTotalImage )
		else:
			pathToSaveImageHH = os.path.join(TARGET_DIRECTORY_, pathToSaveImageHH )
			pathToSaveImageHV = os.path.join(TARGET_DIRECTORY_, pathToSaveImageHV )
			pathToSaveTotalImage = os.path.join(TARGET_DIRECTORY_, pathToSaveTotalImage )

		cv2.imwrite(pathToSaveImageHH, imageHH_)
		cv2.imwrite(pathToSaveImageHV, imageHV_)
		cv2.imwrite(pathToSaveTotalImage, imageTotal_)


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: DrawRadarImage  $(DataDir)  $(TargetDir)")
		print("Example: DrawRadarImage.py  data/train.json  train_peak")

	else:
		STATOI_DATA__PATH = sys.argv[1]
		TARGET_DIR = sys.argv[2]

		dataReader = StatoilDataReader()
		listOfRadarImage = dataReader.Read(STATOI_DATA__PATH)
		dataDrawer = DataDrawer()
		dataDrawer.Draw(listOfRadarImage, TARGET_DIR)

		summaryFilePathName = os.path.join(TARGET_DIR, "Summary.txt")
		print("Write Summary to " + summaryFilePathName)
		with open(summaryFilePathName, 'w') as summaryFile:
			summaryFile.write("FileID \t\t Category \t\t Angle\n")
			for each in listOfRadarImage:
				category = "?"
				angle = "NA"
				if each.hasAnswer:
					if each.isIceberg:
						category = "ice"
					else:
						category = "ship"
				if each.hasAngle:
					angle = "{0:.2f}".format(each.angle)
				info = each.name + "\t\t" + str(category) + "\t\t" + angle + "\n"
				summaryFile.write(info)
			
