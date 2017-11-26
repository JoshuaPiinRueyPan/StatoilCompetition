import tensorflow as tf

def ConvLayer(x, W, b, strides=1, padding='SAME', name=None):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
	x = tf.nn.bias_add(x, b)

	layerName = name
	if layerName is None:
		layerName = 'layer: s='+str(strides)
	tf.summary.histogram(layerName + '/weights', W)
	return tf.nn.relu(x, name = name)

def AlexNorm(inputTensor, lsize=4, name=None):
	return tf.nn.lrn(inputTensor, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)

def MaxPoolLayer(x, kernelSize=2, name=None):
	return tf.nn.max_pool(x, ksize=[1, kernelSize, kernelSize, 1],
				 strides=[1, kernelSize, kernelSize, 1], padding='SAME', name=name)


def AvgPoolLayer(x, kernelSize, name=None):
	return tf.nn.avg_pool(x, ksize=[1, kernelSize, kernelSize, 1],
				 strides=[1, kernelSize, kernelSize, 1],
				 padding='SAME',
				 name='avg_pool')
