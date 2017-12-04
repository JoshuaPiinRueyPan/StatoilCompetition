#!/usr/bin/python
import sys
import cv2
import os
from src.RadarImage import *
from src.DataAugmentation import *

class DataShow:
	def ShowAugmentation(self, listOfRadarImage_):
		for eachRadarImage in listOfRadarImage_:
			self.eachRadarImage(eachRadarImage)
			cv2.imshow("AugumentData", self.imageAll)
			userResponse = cv2.waitKey(0)
			if userResponse == ord('n'):
				pass
			elif userResponse == ord('q'):
				cv2.destroyAllWindows()
				exit()

	def __init__(self):
		self.dataAugmentation = DataAugmentation()
		self.imageAll = np.zeros((300*2, 300*2,3),dtype=np.uint8)

	def resizeImageToTargetSize(self, image_):
		return cv2.resize(image_, (300,300))

	def eachRadarImage(self, radarImage_):
		imageName = radarImage_.name

		imageHH = radarImage_.GetImageHH()
		imageHH = self.resizeImageToTargetSize(imageHH)

		imageHV = radarImage_.GetImageHV()
		imageHV = self.resizeImageToTargetSize(imageHV)

		imageHHwirhAugmentation = self.dataAugmentation.Augment(imageHH)
		imageHVwirhAugmentation = self.dataAugmentation.Augment(imageHV)

		self.imageAll[:300, :300] = imageHH
		self.imageAll[:300, 300:] = imageHHwirhAugmentation
		self.imageAll[300:, :300] = imageHV
		self.imageAll[300:, 300:] = imageHVwirhAugmentation


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
			
