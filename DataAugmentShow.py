#!/usr/bin/python
import sys
import cv2
import os
from src.RadarImage import *
from src.DataAugmentation import *

class DataShow:
	def ShowAugmentation(self, listOfRadarImage_):
		NUMBER_OF_DATA = len(listOfRadarImage_)
		index = 0
		while True:
			currentImage = listOfRadarImage_[index]
			self.drawEachRadarImage(currentImage)
			cv2.imshow("AugumentData", self.canvas)
			userResponse = cv2.waitKey(0)
			if userResponse == ord('n'):
				index += 1
				self.canvas.fill(0)

			elif userResponse == ord('p'):
				index -= 1
				self.canvas.fill(0)

			elif userResponse == ord('q'):
				cv2.destroyAllWindows()
				exit()

			if index >= NUMBER_OF_DATA:
				index = 0

			elif index < 0:
				index = NUMBER_OF_DATA - 1


	def __init__(self):
		self.dataAugmentation = DataAugmentation()
		self.canvas = np.zeros((500*2, 500*2),dtype=np.uint8)

	def resizeImageToTargetSize(self, image_):
		return cv2.resize(image_, (500,500))

	def drawEachRadarImage(self, radarImage_):
		totalImage = radarImage_.GetTotalImage()

		augmentedTotalImage, operations = self.dataAugmentation.Augment(totalImage)

		totalImage = (totalImage * 255.0).astype(np.uint8)
		totalImage = self.resizeImageToTargetSize(totalImage)

		augmentedTotalImage = (augmentedTotalImage * 255.0).astype(np.uint8)
		augmentedTotalImage = self.resizeImageToTargetSize(augmentedTotalImage)

		imageHH = totalImage[:, :, 0]
		imageHV = totalImage[:, :, 1]

		augmentedImageHH = augmentedTotalImage[:, :, 0]
		augmentedImageHV = augmentedTotalImage[:, :, 1]

		WHITE = (255, 255, 255)
		# draw imageHH
		self.canvas[:500, :500] = imageHH
		cv2.putText(self.canvas, "HH", (250, 450), cv2.FONT_HERSHEY_SIMPLEX,
			    fontScale=0.8, color=WHITE, thickness=2)

		# draw aug. imageHH
		self.canvas[:500, 500:] = augmentedImageHH
		cv2.putText(self.canvas, "aug. HH", (750, 400), cv2.FONT_HERSHEY_SIMPLEX,
			    fontScale=0.8, color=WHITE, thickness=2)
		cv2.putText(self.canvas, operations, (520, 450), cv2.FONT_HERSHEY_SIMPLEX,
			    fontScale=0.6, color=WHITE, thickness=1)

		# draw imageHV
		self.canvas[500:, :500] = imageHV
		cv2.putText(self.canvas, "HV", (250, 900), cv2.FONT_HERSHEY_SIMPLEX,
			    fontScale=0.8, color=WHITE, thickness=2)

		# draw aug. imageHV
		self.canvas[500:, 500:] = augmentedImageHV
		cv2.putText(self.canvas, "aug. HV", (750, 900), cv2.FONT_HERSHEY_SIMPLEX,
			    fontScale=0.8, color=WHITE, thickness=2)
		cv2.putText(self.canvas, operations, (520, 950), cv2.FONT_HERSHEY_SIMPLEX,
			    fontScale=0.6, color=WHITE, thickness=1)

		cv2.line(self.canvas, (500, 50), (500, 1000), color=WHITE, thickness=2)
		cv2.line(self.canvas, (0, 500), (1000, 500), color=WHITE, thickness=2)

		self.writeTitleToImage(self.canvas, radarImage_)

	def writeTitleToImage(self, image_, radarImage_):
		categoryInfo = "?"
		if radarImage_.hasAnswer:
			if radarImage_.isIceberg:
				categoryInfo = "ice"
			else:
				categoryInfo = "ship"
		angleInfo = "NA"
		if radarImage_.hasAngle:
			angleInfo = "{0:.2f}".format(radarImage_.angle)
		imageInfo =  radarImage_.name + ";  "  + categoryInfo + ";  angle: " + angleInfo
		cv2.putText(image_, imageInfo, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: DrawRadarImage  $(DataDir)")
		print("Example: DrawRadarImage.py  data/train.json")
		print("n: next image, q: quit ! ")
	else:
		STATOI_DATA__PATH = sys.argv[1]

		dataReader = StatoilDataReader()
		listOfRadarImage = dataReader.Read(STATOI_DATA__PATH)
		dataShow = DataShow()
		dataShow.ShowAugmentation(listOfRadarImage)
			
