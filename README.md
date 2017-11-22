# Statoil Iceberg Recognition
This package provide some basic utilities (such as data loading, data drawing, accuracy calculation... etc) that facilitate the design of network for Statoil Iceberg Recognition.

## Requirements
1. Tensorflow
2. opencv

## Architecture
   To avoid the conflict between users, this package use the following architecture:
1. The Train.py has fixed procedures and some customized settings can be editted in settings/TrainingSettings.py in which will be ignored by Git so that the user will not get conflict if they both editted certain variables.

2. The Train.py will train the IceNet in src/IceNet.py.  For the Statoil Iceberg Competition, the input and the loss layer is fixed.  Therefore, we only need to change the net body (called the Subnet here) and make sure it return a tensor with the shape (batchSize, numberOfCategories).

3. The src/IceNet.py will call SubnetFactory in settings/SubnetSettings.py.  If you want to change the subnet, you simply just return which subnet you want.

## Setup
1. At the first time, you should copy settings/*.example to settings*.py as follows:
	```Shell
	$ cp settings/OutputSettings.example  settings/OutputSettings.py
	$ cp settings/SubnetSettings.example  settings/SubnetSettings.py
	$ cp settings/TrainingSettings.example  settings/TrainingSettings.py
	```
  As illustrate above, the files in settings/ will not upload to avoid conflict in version.  You need to write your customized settings in settings/*.py.  You can refer to settings/*.example to see how to write settings.


## How to Customize Subnet
  You can refer to src/subnet/AlexnetTiny.py as an example.  The procedure is listed below:
1. Create your own subnet in src/subnet/mySubnet.py
2. 
class AlexnetTiny(SubnetBase):
	def __init__(self, isTraining_, inputImage_, inputAngle_, groundTruth_):
		self.isTraining = isTraining_
		self.inputImage = inputImage_
		self.inputAngle = inputAngle_
		self.groundTruth = groundTruth_

	def Build(self):
		weights, biases = self.buildNetVariables()
		return self.buildNetBody(weights, biases)


## Note
If you're using python3, execute the program like:
	```Shell
	$ PYTHONPATH=.  python3  Train.py
	```
