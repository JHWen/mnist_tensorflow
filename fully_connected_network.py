import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# input size and output size
input_size = 784
hidden_size = 500
output_size = 10

learning_rate_base = 0.8
learning_rate_decay = 0.99

batch_size = 100

regularization_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99

model_save_path = './model_path/'
model_name = 'fully_connected_model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_size], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, output_size], name='y-input')
    global_step = tf.Variable(0, trainable=False)

    # hidden layer
    hidden_output = tf.contrib.layers.fully_connected(inputs=x, num_outputs=hidden_size, activation_fn=tf.nn.relu)

    # output layer
    y = tf.contrib.layers.fully_connected(inputs=hidden_output, num_outputs=output_size, activation_fn=None)

    # loss calculate
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(input=y_, dimension=1)))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # train op
    train_step = tf.train.GradientDescentOptimizer(learning_rate_base).minimize(loss=cross_entropy,
                                                                                global_step=global_step)
    # validation
    correct_predict = tf.equal(tf.arg_max(input=y, dimension=1), tf.arg_max(input=y_, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    with tf.Session() as sess:
        # init parameters
        tf.global_variables_initializer().run()
        # validate data
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # test data
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # saver
        saver = tf.train.Saver()

        # training
        for i in range(training_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy is %g' % (i, validate_acc))
                saver.save(sess=sess, save_path=os.path.join(model_save_path, model_name), global_step=global_step)

            xi, yi = mnist.train.next_batch(batch_size)
            _, loss_ = sess.run([train_step, cross_entropy], feed_dict={x: xi, y_: yi})
            # print(loss_)
            if i % 100 == 0:
                print('train step %d, loss %g' % (i, loss_))

        # test
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy is %g' % (training_steps, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets(train_dir='./data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
