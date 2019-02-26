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
    shuff_idx = np.random.randint(batch_xs.shape[0], size=10)
    labels, feature_tensor = sess.run([model, L4],
                                     feed_dict={X: batch_xs,
                                                Y: batch_ys,
                                                keep_prob:1})

    ##########################################################################
    # TODO: Mission 5)                                                       #
    #   From the trained model, you could extract class activation map.      #
    #   Pick a set of feature tensor and also pick a matching label weight ! #
    #   You should pay attention to index of tensor and label.               # 
    ########################################################################## 
    # TODO 5-1)                                                              #
    # We get random index in 'shuff_idx'                                     #
    # You should pick a matching tensor using 'shuff_idx'                    #
    ##########################################################################
    picked = 
    # And pick a matching labels using 'shuff_idx'
    picked_labels = np.asarray(labels[shuff_idx], dtype=np.float32)
    picked_labels = 
    ########################################################################## 

    prediction = np.zeros((10, 10), dtype=np.float32)
    prediction[np.arange(10), picked_labels] = 1.0



    ########################################################################## 
    # TODO 5-2)
    # From the class-weight(W5), pick a weight vector corresponding to 'picked_labels' 
    # Hint: .eval(session=sess) would make tensor to 
    matching_weight = 
    mat = np.reshape(matching_weight, (10, 1, 1, 1024))
    output = 


    ########################################################################## 


    fig = plt.figure()
    cnt = 1
    for i in range(10):
        idx = shuff_idx[i]
        if i == 5:
            cnt = 6
        subplot = fig.add_subplot(4, 5, i + cnt)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('%d' % np.argmax(picked_labels[i])) # prediction
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
        subplot.imshow(colored)#cmap=plt.cm.gray_r)
        subplot = fig.add_subplot(4,5,i+cnt+5)
        subplot.imshow(img)

    plt.savefig('cam_result.png')
    plt.show()
    
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


# TODO: Mission 1)  ##############################################################################
#   You are going to build a feature extractor which is for Class Activation Map.                #
#   MNIST dataset is not that large dataset, so you can stack a few layers.                      # 
#   First, define a placeholder for MNIST input and labels.                                      #
#   You may care about output dimension !                                                        #
#   Hint 1) tf.Varaible, tf.random_normal, tf.nn.relu, tf.nn.conv2d ...                          #
#   Hint 2) The size of MNIST data is (28, 28) and has 10 labels ( 0 ~ 9 )                       # 
##################################################################################################


X =
Y = 

W1 = 
L1 =
L1 =

... 



# TODO: Mission 2)  ##############################################################################
#   The core idea of the Class Activation Map(CAM) is Global Average Pooling (GAP)               # 
#   Global Average Pooling is the operation aggregating tensor score in channel-wise.            #
#   You should make GAP layer and complete the following architecture. Refer to figure on slide. # 
#   'model' parameter is for your model prediction.                                              # 
##################################################################################################






# TODO: Mission 3)  ##############################################################################
#   Define an opitmizer for training !                                                           #
#   You should take 2-steps as follows.                                                          # 
#       1) Calculate cost with cross-entropy-loss function.                                      #
#       2) Choose an optimizer for minimizing your cost.                                         # 
#   Hint : tf.reduce_mean, tf.nn. ( - ) ,                                                        #
##################################################################################################








# TODO: Mission 4)  ##############################################################################
#   The system without regularization loss will perform good on classification.                  #
#   But the output of CAM is not that good.                                                      #
#   Let's add a regularization term!                                                             #
#   You may use l2-loss term.                                                                    #
#   Hint: tf.nn.l2_loss                                                                          #   
##################################################################################################




init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
sess.run(init)


# TODO: 
batch_size = 10
total_batch = int(mnist.train.num_examples / batch_size)

# Save the model
saver = tf.train.Saver()

for epoch in range(5):
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
#cam()

