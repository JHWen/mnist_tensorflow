from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

# LeNet-5 implementation

# modify image input size from 784 to 28*28*1
IMAGE_SIZE = 28
NUM_CHANNELS = 1

INPUT_SIZE = 784
LABEL_NUM = 10
BATCH_SIZE = 50
LEARNING_RATE = 0.0005
TRAINING_STEPS = 30000

# conv2d size
# conv layer1
conv1_size = 5
conv1_deep = 32

# conv layer2
conv2_size = 5
conv2_deep = 64

# fully connected layer
fc_size = 512
drop_out = 0.5

# model save
MODEL_SAVE_PATH = './model_path/'
MODEL_NAME = 'model_cnn.ckpt'

# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0


def inference(input_tensor, keep_prob):
    """

    :param input_tensor: reshaped x-input
    :param keep_prob: train or test
    :return: logits
    """
    # layer1 conv1 5*5*32
    # batch_size*28*28*1 ->  batch_size*28*28*32  (all 0 padding)
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(name='weight', shape=[conv1_size, conv1_size, NUM_CHANNELS, conv1_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(name='bias', shape=[conv1_deep], initializer=tf.constant_initializer(value=0.0))
        conv1 = tf.nn.conv2d(input=input_tensor, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # layer2 pool1 5*5 stride 2*2
    # batch_size*28*28*32 -> batch_size*14*14*32
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(value=relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # layer3 conv2 5*5*64
    # batch_size*14*14*32 -> batch_size*14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(name='weight', shape=[conv2_size, conv2_size, conv1_deep, conv2_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(name='bias', shape=[conv2_deep], initializer=tf.constant_initializer(value=0.0))
        conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # layer4 pool2 size 2*2 strid 2*2
    # batch_size*14*14*64 -> 7*7*64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(value=relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # calculate pool2 output shape
    pool2_shape = pool2.get_shape().as_list()
    # 7*7*64
    fc1_input_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]

    # batch_size*3136
    reshaped_fc1_input = tf.reshape(tensor=pool2, shape=[-1, fc1_input_size])

    # layer5 fully connected network
    # batch_size*3136 -> batch_size*512
    with tf.variable_scope('layer5-fc1'):
        fc1 = tf.contrib.layers.fully_connected(inputs=reshaped_fc1_input, num_outputs=fc_size,
                                                activation_fn=tf.nn.relu)

        fc1_dropout = tf.nn.dropout(x=fc1, keep_prob=keep_prob)

    # batch_size*512 -> batch_size*10
    with tf.variable_scope('layer6-fc2'):
        logits = tf.contrib.layers.fully_connected(inputs=fc1_dropout, num_outputs=LABEL_NUM, activation_fn=None)

    return logits


def loss_fn(logits, y):
    """

    :param logits:
    :param y:
    :return:
    """
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    return cross_entropy


def train(mnist):
    # define variables
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='y-input')
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    # adjust matrix to batch_size*image_size*image_size*channels(depth)
    reshaped_x = tf.reshape(tensor=x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    logits = inference(input_tensor=reshaped_x, keep_prob=keep_prob)

    # loss calculate
    loss = loss_fn(logits, y_)

    # train op
    # train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step)
    # test
    correct_predict = tf.equal(tf.arg_max(input=logits, dimension=1), tf.arg_max(input=y_, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    with tf.Session() as sess:
        # init parameters
        tf.global_variables_initializer().run()

        # test data
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels,
            keep_prob: 1.0
        }

        # validate data
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels,
            keep_prob: 1.0
        }

        saver = tf.train.Saver()
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # validate_acc = evaluate(sess, validate_feed)
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy is %g' % (i, validate_acc))
                # saver.save(sess=sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

            xi, yi = mnist.train.next_batch(BATCH_SIZE)
            _, loss_ = sess.run([train_op, loss], feed_dict={x: xi, y_: yi, keep_prob: 0.5})
            if i % 100 == 0:
                print('train step %d, loss %g' % (i, loss_))
        # test
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy is %g' % (TRAINING_STEPS, test_acc))


def evaluate(sess, eval_feed):
    """

    :param sess:
    :param eval_feed:
    :return:
    """
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, LABEL_NUM], name='y-input')

    # adjust matrix to batch_size*image_size*image_size*channels(depth)
    reshaped_x = tf.reshape(tensor=x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    logits = inference(input_tensor=reshaped_x, train=False)

    correct_predict = tf.equal(tf.arg_max(input=logits, dimension=1), tf.arg_max(input=y_, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    acc = sess.run(accuracy, feed_dict=eval_feed)
    return acc


def main(argv=None):
    """

    :param argv:
    :return:
    """
    mnist = input_data.read_data_sets(train_dir='./data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
