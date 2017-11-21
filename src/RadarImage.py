import numpy as np
import cv2
import json

DATA_WIDTH = 75
DATA_HEIGHT = 75

def ScaleArrayToNormalImageValue(array, maxValue_, minValue_):
	scale = 255. / (maxValue_ - minValue_)
	array = (array - minValue_) * scale
	return array.astype(np.uint8)


class RadarImage:
	def NormalizeImage(self, maxHH_, minHH_, maxHV_, minHV_):
		self.dataHH = ScaleArrayToNormalImageValue(self.dataHH, maxHH_, minHH_)
		self.dataHV = ScaleArrayToNormalImageValue(self.dataHV, maxHV_, minHV_)
		self.isNormalized = True

	def GetImageHH(self):
		if not self.isNormalized:
			raise UserWarning("You should Normalize RadarImage before you use it!")
		else:
			singleChannelImage = np.reshape(self.dataHH, (DATA_WIDTH, DATA_HEIGHT))
			normalImage = cv2.cvtColor(singleChannelImage, cv2.COLOR_GRAY2BGR)
			return normalImage

	def GetImageHV(self):
		if not self.isNormalized:
			raise UserWarning("You should Normalize RadarImage before you use it!")
		else:
			singleChannelImage = np.reshape(self.dataHV, (DATA_WIDTH, DATA_HEIGHT))
			normalImage = cv2.cvtColor(singleChannelImage, cv2.COLOR_GRAY2BGR)
			return normalImage

	def GetTotalImage(self):
		if not self.isNormalized:
			raise UserWarning("You should Normalize RadarImage before you use it!")
		else:
			stuffValue = 0
			if self.hasAngle:
				stuffValue = np.array(self.angle)
			else:
				stuffValue = np.array(0)

			imageHH = np.reshape(self.dataHH, (DATA_WIDTH, DATA_HEIGHT))
			imageHV = np.reshape(self.dataHV, (DATA_WIDTH, DATA_HEIGHT))
			imageAngleChannel = np.tile(stuffValue, (DATA_WIDTH, DATA_HEIGHT)).astype(np.uint8)
			
			temp = np.stack((imageHH, imageHV, imageAngleChannel))
			'''
			    After stacked, the shape will be: (3, 75, 75),
			    we need to transpose it back.
			'''
			return np.transpose(temp, (1, 2, 0))

	def __init__(self, statoilData_):
		self.name = statoilData_["id"]
		try:
			self.isIceberg = statoilData_["is_iceberg"]
			self.hasAnswer = True
		except:
			self.isIceberg = -1
			self.hasAnswer = False

		self.dataHH = np.array( statoilData_["band_1"] )
		self.dataHV = np.array( statoilData_["band_2"] )
		self.angle = statoilData_["inc_angle"]
		if (self.angle == "nan")or(self.angle=="na"):
			self.angle = -1
			self.hasAngle = False
		else:
			self.hasAngle = True

		self.isNormalized = False
	

class StatoilDataReader:
	def Read(self, STATOI_DATA__PATH_):
		rawData = None
		with open(STATOI_DATA__PATH_,"r") as dataFile:
			rawData = json.load(dataFile)

		listOfRadarImage = self.extractRadarImageFromData(rawData)
		maxHH, minHH, maxHV, minHV = self.getExtremaForRadarImageList(listOfRadarImage)
		self.normalizeRadarImageList(listOfRadarImage, maxHH, minHH, maxHV, minHV)
		return listOfRadarImage

	def extractRadarImageFromData(self, listOfData_):
		listOfRadarData = []
		for eachData in listOfData_:
			currentRadarImage = RadarImage(eachData)
			listOfRadarData.append(currentRadarImage)

		return listOfRadarData

	def getExtremaForRadarImageList(self, listOfRadarImage_):
		listOfMaxHH = []
		listOfMinHH = []

		listOfMaxHV = []
		listOfMinHV = []
		for eachImage in listOfRadarImage_:
			listOfMaxHH.append( np.amax(eachImage.dataHH) )
			listOfMinHH.append( np.amin(eachImage.dataHH) )

			listOfMaxHV.append( np.amax(eachImage.dataHV) )
			listOfMinHV.append( np.amin(eachImage.dataHV) )

		maxHH = max(listOfMaxHH)
		minHH = min(listOfMinHH)

		maxHV = max(listOfMaxHV)
		minHV = min(listOfMinHV)

		print("\t\t(maxHH, minHH, maxHV, minHV) = ("+str(maxHH)+", "+str(minHH)+", "+str(maxHV)+", "+str(minHV)+")")
		return maxHH, minHH, maxHV, minHV

	def normalizeRadarImageList(self, listOfRadarImage_, maxHH_, minHH_, maxHV_, minHV_):
		for eachImage in listOfRadarImage_:
			eachImage.NormalizeImage(maxHH_, minHH_, maxHV_, minHV_)

	def __init__(self):
		pass

