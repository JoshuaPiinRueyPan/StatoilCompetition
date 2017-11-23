import tensorflow as tf

def ConvLayer(name, x, W, b, strides=1, padding='SAME'):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x, name = name)

def MaxPoolLayer(name, x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def AlexNorm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


def AvgPoolLayer(name, x, kernelSize):
	return tf.nn.avg_pool(x, ksize=[1, kernelSize, kernelSize, 1],
				 strides=[1, kernelSize, kernelSize, 1],
				 padding='VALID',
				 name='avg_pool')
