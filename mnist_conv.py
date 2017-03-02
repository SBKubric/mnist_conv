import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib._cm import _binary_data as cm_binary
import numpy as np
from skimage import data
from skimage.transform import resize

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist_samples = mnist.train.images[:32, :].transpose().reshape(28, 28, -1)

fig = plt.figure(figsize=(10, 6))
for j in range(mnist_samples.shape[2]):
    ax = fig.add_subplot(4, 8, j+1)
    ax.imshow(mnist_samples[:, :, j], cmap=matplotlib.cm.binary, interpolation='none')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.show()

with tf.name_scope('inputs'):
    x = tf.placeholer(tf.float32, [None, 784], name='x') # None - batch size, 784 - 28x28 (image vector)
    y = tf.placeholder(tf.float32, [None, 10], name='y') # Output placeholder
    x_image = tf.reshape(x, [-1, 28, 28, 1], name='image') # differences: we created a graph operation -1 - we dont know размерность, 1 - чб картинка


with tf.name_scope('conv-1'):
    with tf.name_scope('params'):
        W_conv_0 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='W')  # нормальное распределение с отклонением 0.1б
        # 4-мерным тензором (два ядра 5х5, 1 картинка, 32 feature-map'a
        b_conv_0 = tf.Variable(tf.constant(0.1, shape=[32]), name='b')  # 32 featurmap - 32 числа смещения (тк сверточная нейронка)
    # srides - шаг свёртки
    conv_0_logit = tf.add(tf.nn.conv2d(x_image, W_conv_0, strides=[1, 1, 1, 1], padding='SAME'), b_conv_0, name='logit')
    h_conv_0 = tf.nn.relu(conv_0_logit, name='relu')

with tf.name_scope('pooling-1'):
    h_pool_0 = tf.nn.max_pool(h_conv_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('conv-layer-2'):
    with tf.name_scope('params'):
        W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='W')  # 32 input feature maps - 64 output feature maps
        b_conv_1 = tf.Variable(tf.constant(0.1, shape=[64]), name='b')
    conv_1_logit = tf.add(tf.nn.conv2d(x_image, W_conv_1, strides=[1, 1, 1, 1], padding='SAME'), b_conv_1, name='logit')
    h_conv_1 = tf.nn.relu(conv_1_logit, name='relu')

with tf.name_scope('pooling-2'):
    h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('fc-1'):
    h_pool_1_flat = tf.reshape(h_pool_1, [-1, 7*7*64], name='flatten')  # в тф по умолчанию
    # стоит паддинг такой, чтобы входная картинка после convolution не менялась, а тк было 2 пулинга,
    # то размер картинки уменьшился в 4 раза и стал 7х7
    with tf.name_scope('params'):
        W_fc_0 = tf.Variable(tf.truncated_normal([7*7*64, 256], stddev=0.1), name='W')
        b_fc_0 = tf.Variable(tf.constant(0.1, shape=[256]), name='b')
    fc_0_logit = tf.add(tf.matmul(h_pool_1_flat, W_fc_0), b_fc_0, name='logit')
    h_fc_0 = tf.nn.relu(fc_0_logit, name='relu')

with tf.name_scope('fc-2'):
    with tf.name_scope('params'):
        W_fc_1 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1), name='W')
        b_fc_1 = tf.Variable(tf.constant(0.1, shape=[10]), name='b')
    logit_out = tf.add(tf.matmul(h_fc_0, W_fc_1), b_fc_1, name='logit')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit_out, y), name='loss')
loss_summary = tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logit_out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

# ADJUST
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth=True
config.gpu_options.allocator_type = 'BFC'

sess = tf.Session(config=config)

sess.run(init)

# ADJUST
summary_writer = tf.summary.FileWriter('.logs/sgd', sess.graph)
# summary_writer = tf.summary.FileWriter('.logs/adam/', sess.graph)

for i in range(4000):

    batch_xs, batch_ys = mnist.train.next_batch(64)
    _, loss_s = sess.run([train_step, loss_summary], feed_dict={x: batch_xs, y: batch_ys})

    if i % 10 == 0:
        summary_writer.add_summary(loss_s, i)

    if i % 200 == 0:
        test_xs, test_ys = mnist.test.next_batch(128)
        acc, acc_s = sess.run([accuracy, accuracy_summary], feed_dict={x: test_xs, y: test_ys})
        summary_writer.add_summary(acc_s, i)
        print('[{}] Accuracy: {}'.format(i, acc))

    print('Final accuracy : {}'.format(sess.run(accuracy, feed_dict={
        x: mnist.test.images,
        y: mnist.test.labels,
    })))

    idx = random.randint(0, mnist.test.images.shape[0])
    sample = mnist.test.images[idx, :][np.newaxis, ...]
    prediction = sess.run(tf.nn.softmax(logit_out), feed_dict={x: sample})
    ans = np.argmax(prediction)

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(sample.transpose().reshape(28, 28), cmap=matplotlib.cm.binary, interpolation='none')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

    ax = fig.add_subplot(1, 2, 2)
    bar_list = ax.bar(np.arange(10), prediction[0], align='center')
    bar_list[ans].set_color('g')
    ax.set_xticks(np.arange(10))
    ax.set_xlim([-1, 10])
    ax.grid('on')

    plt.show()
