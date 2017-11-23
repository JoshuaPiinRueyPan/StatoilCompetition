# Statoil Iceberg Recognition Development Platform
This package provide some basic utilities (such as data loading, data drawing, accuracy calculation... etc) that facilitate the design of network for Statoil Iceberg Competition.

## Requirements
1. Tensorflow
2. opencv

## Architecture
   To avoid the conflicts between users, this package has the following architecture:
1. The Train.py has fixed procedures and some customized settings can be adjusted in settings/TrainingSettings.py in which will be ignored by Git so that the user will not get conflict if they both edit certain variables.

2. The Train.py will train the IceNet in src/IceNet.py.  For the Statoil Iceberg Competition, the input and the loss layer is fixed.  Therefore, the architecture of IceNet can be fixed.  We only need to customize its net body (called the Subnet here) in which you can define network such as AlexNet or ResNet like architecture.  And make sure it return a tensor as the shape (batchSize, numberOfCategories).

3. The src/IceNet.py will call SubnetFactory in settings/SubnetSettings.py.  If you want to change the subnet, you simply just return which subnet you want.


## Setup
1. At the first time, you should copy settings/\*.example to settings/\*.py as follows:
	```Shell
	$ cp settings/OutputSettings.example  settings/OutputSettings.py
	$ cp settings/SubnetSettings.example  settings/SubnetSettings.py
	$ cp settings/TrainingSettings.example  settings/TrainingSettings.py
	$ cp settings/TestSettings.example  settings/TestSettings.py
	```
  As illustrate above, the files in settings/ will not upload to avoid conflict in version.  You need to write your customized settings in settings/\*.py.  You can refer to settings/\*.example to see how to write settings.


## How to Customize Subnet
  You can refer to src/subnet/AlexnetTiny.py as an example.  The procedure is listed below:
1. Create your own subnet in src/subnet/mySubnet.py

2. Your customized subnet should have the following structre:
	```
	class MySubNet(SubnetBase):
		def __init__(self, isTraining_, inputImage_, inputAngle_, groundTruth_):
		def Build(self):
	```
    Note: isTraining_, inputImage_, inputAngle_, groundTruth_ is the placeholder.

3. Edit SubnetFactory in settings/SubnetSettings.py to return your customized subnet so that Train.py will use your customized subnet to train the new model.


## Utilities
1. Train.py: Train the network.  The model will be saved in temp/
2. DrawRadarImage.py:  The original data is in json format.  This tool can unpack the json file and draw the data as RGB images.


## Note
If you're using python3, execute the program like:
	```Shell
	$ PYTHONPATH=.  python3  Train.py
	```
