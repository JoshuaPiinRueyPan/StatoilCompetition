import tensorflow as tf
import matplotlib.pyplot as plt
import json
import numpy as np
from random import shuffle

learning_rate = 0.0001
training_iters = 20000
batch_size = 100
display_step = 1000

n_input = 75*75*2
n_classes = 2
dropout = 0.5

NUMBER_OF_VALIDATION = 200

with open("train.json","r") as f:
    load_dict = json.load(f)
    train_dict = load_dict[NUMBER_OF_VALIDATION:]
    validation_dict = load_dict[:NUMBER_OF_VALIDATION]

maxnumber1 = 34.5749
minnumber1 = -45.5944
maxnumber2 = 20.1542
minnumber2 = -45.6555

shuffle(train_dict)

def get_batch(load_dict, batch_size, step):
    batch_x_list = []
    batch_y_list = []
    batch_x_angle_list = []
    for i in range(batch_size):
        idx = ((step-1)*batch_size+i) % len(load_dict)
        x_list = []            
        for j in range(75*75):
            x_list.append((load_dict[idx]['band_1'][j]-minnumber1)/(maxnumber1-minnumber1))
            x_list.append((load_dict[idx]['band_2'][j]-minnumber2)/(maxnumber2-minnumber2))
            #x_list.append(load_dict[idx]['band_1'][j]/100.+0.5)
            #x_list.append(load_dict[idx]['band_2'][j]/100.+0.5)
        x_array = np.array(x_list)
        batch_x_list.append(x_array)

        if load_dict[idx]['is_iceberg'] == 1:
            y_list = np.array([0.,1.])
        elif load_dict[idx]['is_iceberg'] == 0:
            y_list = np.array([1.,0.])
        batch_y_list.append(y_list)

        #x_angle = []
        if load_dict[idx]['inc_angle'] != 'na':
            batch_x_angle_list.append(np.array([load_dict[idx]['inc_angle']], np.float32))
        else:
            batch_x_angle_list.append([-1])#39.26
    #print(batch_x_list)

    return (np.array(batch_x_list), np.array(batch_x_angle_list), np.array(batch_y_list))

def convlayer(name, x, W, b, strides=1, padding='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name = name)

def maxpool(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)

x_img = tf.placeholder(tf.float32, [None, n_input])
x_angle = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'convW1': tf.Variable(tf.random_normal([3, 3, 2, 8], stddev=0.01)),
    'convW2': tf.Variable(tf.random_normal([3, 3, 8, 16], stddev=0.01)),
    'convW3': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
    'convW4': tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01)),
    'fcW1'  : tf.Variable(tf.random_normal([5*5*16+1, 128], stddev=0.01)),
    'fcW2'  : tf.Variable(tf.random_normal([128, 128], stddev=0.01)),
    'output': tf.Variable(tf.random_normal([128, n_classes], stddev=0.01)),
}

biases = {
    'convb1': tf.Variable(tf.random_normal([8], stddev=0.01)),
    'convb2': tf.Variable(tf.random_normal([16], stddev=0.01)),
    'convb3': tf.Variable(tf.random_normal([16], stddev=0.01)),
    'convb4': tf.Variable(tf.random_normal([16], stddev=0.01)),
    'fcb1'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
    'fcb2'  : tf.Variable(tf.random_normal([128], stddev=0.01)),
    'output'  : tf.Variable(tf.random_normal([n_classes], stddev=0.01)),
}

def myNet(x_img, x_angle, weights, biases, dropout):
    inputx = tf.reshape(x_img, shape=[-1, 75, 75, 2])

    conv1 = convlayer('conv1', inputx, weights['convW1'], biases['convb1'])
    pool1 = maxpool('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, lsize=4)
    
    conv2 = convlayer('conv2', norm1, weights['convW2'], biases['convb2'])
    pool2 = maxpool('pool2', conv2, k=2)
    norm2 = norm('norm3', pool2, lsize=4)

    conv3 = convlayer('conv3', norm2, weights['convW3'], biases['convb3'])
    pool3 = maxpool('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)

    conv4 = convlayer('conv4', norm3, weights['convW4'], biases['convb4'], padding='VALID')
    pool4 = maxpool('pool4', conv4, k=2)
    norm4 = norm('norm4', pool4, lsize=4)

    pool4reshape = tf.reshape(norm4, [-1, weights['fcW1'].get_shape().as_list()[0]-1])
    concat = tf.concat(axis=1, values=[pool4reshape, x_angle])

    fc1 = tf.add(tf.matmul(concat, weights['fcW1']), biases['fcb1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.reshape(fc1, [-1, weights['fcW2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['fcW2']), biases['fcb2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    output = tf.add(tf.matmul(fc2, weights['output']), biases['output'])
    return output

pred = myNet(x_img, x_angle, weights, biases, keep_prob)

saver = tf.train.Saver()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
correct_pred = tf.reshape(correct_pred, shape=[-1])

accuracy_rate = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    step = 1
    while step < training_iters:
        batch_x, batch_x_angle, batch_y = get_batch(train_dict, batch_size, step)

        sess.run(optimizer, feed_dict={x_img:batch_x, x_angle:batch_x_angle, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            print("step:"+str(step)+"======================================")
            loss, accuracy = sess.run( [cost, accuracy_rate], feed_dict={x_img:batch_x, x_angle:batch_x_angle, y: batch_y, keep_prob: 1.})
            print("train:")
            print("loss: " + str(loss) + ", accuracy: " + str(accuracy)+"\n")
            validation_x, validation_x_angle, validation_y = get_batch(validation_dict, len(validation_dict), 1)
            loss, accuracy = sess.run( [cost, accuracy_rate], feed_dict={x_img:validation_x, x_angle:validation_x_angle, y: validation_y, keep_prob: 1.})
            print("validation:")
            print("loss: " + str(loss) + ", accuracy: " + str(accuracy))
            if step >= 10000:
                saver.save(sess, "save_step"+ str(step) + "/iceberg.ckpt")
        step += 1
        #print("step"+str(step)+" complete!")
    print("Optimization finished!")


"""with tf.Session() as sess:
    sess.run(init)

    step = 1
    batch_x, batch_y = get_batch(load_dict, batch_size, step)

    fc1 = sess.run( fc1, feed_dict={x:batch_x, y: batch_y, keep_prob: 1.})
    print(fc1.shape)
    print("Optimization finished!")"""

#print("Testing Accuracy:", sess.run(accuracy, feed_dict={???????????}))
