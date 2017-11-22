# Statoil Iceberg Recognition
This package provide some basic utilities (such as data loading, data drawing, accuracy calculation... etc) that facilitate the design of network for Statoil Iceberg Recognition.

## Requirements
1. Tensorflow
2. opencv

## Architecture
   To avoid the conflict between users, this package use the following architecture:
1. The Train.py has fixed procedures and some customized settings can be editted in settings/TrainingSettings.py in which will be ignored by Git so that the user will not get conflict if they both editted certain variables.

2. The Train.py will train the IceNet in src/IceNet.py.  For the Statoil Iceberg Competition, the input and the loss layer is fixed.  Therefore, we only need to change the net body (called the Subnet here) and make sure it return a tensor with the shape (batchSize, numberOfCategories).

3. The src/IceNet.py will call SubnetFactory in settings/SubnetSettings.py.  If you want to add your own net, you simply just need
