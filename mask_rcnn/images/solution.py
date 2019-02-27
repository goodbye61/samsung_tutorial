import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pdb
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def plot():
    pdb.set_trace()
    plot_sample = 100
    batch_xs = mnist.test.images[:100].reshape(-1,28,28,1)
    batch_ys = mnist.test.labels[:100]

    labels = sess.run(model,
                      feed_dict={X: batch_xs,
                                 Y: batch_ys,
                                 keep_prob:1})

    shuff_idx = np.random.randint(batch_xs.shape[0], size=10)
    fig = plt.figure()
    for i in range(10):
            idx = shuff_idx[i]
            subplot = fig.add_subplot(2, 5, i + 1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title('%d' % np.argmax(labels[idx])) # prediction
            subplot.imshow(mnist.test.images[idx].reshape((28, 28)),
                          cmap=plt.cm.gray_r)

    plt.show()

def cam():
 
    batch_xs = mnist.test.images[:100].reshape(-1,28,28,1)
    batch_ys = mnist.test.labels[:100]

    labels, feature_tensor = sess.run([model, L4],
                                     feed_dict={X: batch_xs,
                                                Y: batch_ys,
                                                keep_prob:1})

    shuff_idx = np.random.randint(batch_xs.shape[0], size=10)
    picked = feature_tensor[shuff_idx]
    picked_labels = np.asarray(labels[shuff_idx], dtype=np.float32)
    picked_labels = np.argmax(picked_labels, axis=1)

    W = W5.eval(session=sess)
    matching_weight = W[:, picked_labels]
    matching_weight = np.reshape(matching_weight, (10,1,1,1024))
    output = (picked * matching_weight).sum(axis=3)
    
    fig = plt.figure()
    cnt = 1
    for i in range(10):
        idx = shuff_idx[i]
        if i == 5:
            cnt = 6
        subplot = fig.add_subplot(4, 5, i + cnt)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('%d' % np.argmax(prediction[i])) # prediction
        maxer = output[i].max()
        miner = output[i].min()
        cam = (output[i] - miner) / (maxer - miner)
        cam = np.uint8(255*cam)
        cam = cv2.resize(cam, (28,28))
        heated = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        orig_img = mnist.test.images[idx].reshape((28,28))
        colored = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
        img = heated * 0.8 + colored * 0.2
        img = np.asarray(img, dtype=np.uint8)
        subplot.imshow(colored)
        subplot = fig.add_subplot(4,5,i+cnt+5)
        subplot.imshow(img)

    plt.savefig('cam_result.png')
    plt.show()
    

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

W3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L3 = tf.nn.conv2d(L1, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)

W4 = tf.Variable(tf.random_normal([3,3,64,1024], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.relu(L4)
gbl_avg = tf.reduce_mean(L4, axis=[1,2]) # size : [batch, 1, 1, 1024]

W5 = tf.Variable(tf.random_normal([1024, 10], stddev=0.01))
L5 = tf.reshape(gbl_avg, [-1,1024])
model = tf.matmul(L5, W5) 

w1_reg = tf.nn.l2_loss(W1)
w3_reg = tf.nn.l2_loss(W3)
w4_reg = tf.nn.l2_loss(W4)
w5_reg = tf.nn.l2_loss(W5)

reg = (w4_reg + w5_reg) /2.0
beta = 2e-4
learning_rate = 2e-4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y) + beta * reg)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
sess.run(init)

batch_size = 10
total_batch = int(mnist.train.num_examples / batch_size)

# Save the model
saver = tf.train.Saver()

for epoch in range(2):
    total_cost = 0
    print('{} epoch ! '.format(epoch))
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
        #print(cost_val)
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))


is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_test = mnist.test.images.shape[0]
batch_size =  100
total_batch = int(num_test / batch_size) 
idx = 0
test_acc = 0 
for i in range(total_batch):
    batch_xs = mnist.test.images[idx:idx+batch_size].reshape(-1,28,28,1)
    batch_ys = mnist.test.labels[idx:idx+batch_size]
    acc = sess.run(accuracy, feed_dict={X: batch_xs, Y:batch_ys, keep_prob:1})
    test_acc += acc 
    idx += batch_size 

test_acc /= total_batch
print(' The test accuracy is : {} %'.format(test_acc))

#plot()
cam()

