# Statoil Iceberg Recognition Development Platform
This package provide some basic utilities (such as data loading, data drawing, several basic layers... etc) that facilitate the design of network for Statoil Iceberg Competition.

Note:  This project is originally developed [here](https://github.com/waynexhsieh/Statoil-C-CORE-Iceberg-Classifier-Challenge-with-myNet).
After the competition finished, I still maintain this project due to its small size dataset that can varify the design of network
quickly.  However, after several interface changed, I decide to detach it from the original site.

## Requirements
1. Tensorflow
2. opencv

## Architecture
   To avoid the conflicts between users, this package has the following architecture:
1. The Train.py has fixed procedures and some customized settings can be adjusted in settings/TrainingSettings.py in which will be ignored by Git so that the user may not get conflict if they both edit some training variables.

2. The Train.py will train the IceNet in src/IceNet.py.  For the Statoil Iceberg Competition, the input and the loss layer is fixed.  Therefore, the architecture of IceNet can be fixed.  IceNet contains a SubNet in which can be customized by the user.  For example, you can design your customized net as AlexNet or ResNet-like architecture.  And make sure it return a tensor as the shape (batchSize, numberOfCategories).

3. The src/IceNet.py will call SubnetFactory in settings/SubnetSettings.py.  If you want to change the Subnet of the IceNet, you simply just return which subnet you want.


## Setup
1. Download the training set & test set from [Kaggle website](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data).
	Note: You should register the Kaggle as well as join this competition to download the data sets.

2. At the first time, you should copy settings/\*.example to settings/\*.py as follows:
	```Shell
	$ cp settings/DataAugmentSettings.example  settings/DataAugmentSettings.py
	$ cp settings/LayerSettings.example  settings/LayerSettings.py
	$ cp settings/OutputSettings.example  settings/OutputSettings.py
	$ cp settings/SubnetSettings.example  settings/SubnetSettings.py
	$ cp settings/TrainingSettings.example  settings/TrainingSettings.py
	$ cp settings/TestSettings.example  settings/TestSettings.py
	```
  As illustrated above, the files in settings/ will not be uploaded to avoid conflict in version.  You need to write your customized settings in settings/\*.py.  You can refer to settings/\*.example to see how to write settings.

Note: This framework is still under development.  If you get error such as:
	```Shell
		AttributeError: 'module' object has no attribute '\*\*\*'
	```
after you update the project from server, it probabily means we add new settings in settings/*.py.  In this case, you can repeat the procedures listes above (i.e. copy settings/*.example  to  settings/*.py).

3. We assume you put the data sets in data/ and the models in temp/models.  You can edit the settings/ to change the default directory.  For example, if you put the train.json in myDir, you can edit the TrainingSettings.py as:
	```Shell
		TRAINING_SET_PATH_NAME = "myDir/train.json"
	```
so that the program can find your training data.

4. Please refer to settings/*.py to see what else can be customized.


## How to Customize Subnet
  You can refer to src/subnet/AlexnetTiny.py as an example.  The procedure is listed below:
1. Create your own subnet in src/subnet/mySubnet.py

2. Your customized subnet should have the following structre:
	```
	class MySubNet(SubnetBase):
		def __init__(self, isTraining_, currentTrainingSteps_, inputImage_, inputAngle_, groundTruth_):
		def Build(self):
	```
    Note: isTraining_, currentTrainingSteps_, inputImage_, inputAngle_, groundTruth_ is the placeholder.
    Note: The Build() function should return 2 tensors: The first one is the output of your net.  The second is the tensor that update your network (if, for example, your network include BatchNormalization).  The update tensor will be called after back propagation.  If your network do not need to be updated (for example, you haven't use BatchNormalization), you can just return an empty operation (tf.no_op()).


3. Edit SubnetFactory in settings/SubnetSettings.py to return your customized subnet so that Train.py will use your customized subnet to train the new model.  DO NOT directly edit the IceNet.py.


## Utilities
1. Train.py: Train the network.  You can adjust training parameters in settings/TrainingSettings.py .
2. DrawRadarImage.py:  The original data is in json format.  This tool can unpack the json file and draw the data as RGB images.
3. Test.py: If you finish your model and are ready to test its results, this scripts will run your model over all TestSet and generate its prediction which you can submit to Kaggle competition.  You can edit parameters (such as PATH_TO_YOUR_MODEL, PATH_TO_YOUR_TESTSET) in settings/TestSettings.py.


## Note
If you're using python3, execute the program like:
	```Shell
	$ PYTHONPATH=.  python3  Train.py
	```
