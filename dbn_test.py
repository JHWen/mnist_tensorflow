from tensorflow.examples.tutorials.mnist import input_data
from rbm import RBM
from nn import NN

if __name__ == '__main__':
    # Loading in the mnist data
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    RBM_hidden_sizes = [500, 200, 50]  # create 4 layers of RBM with size 785-500-200-50
    # Since we are training, set input as training data
    inpX = trX
    # Create list to hold our RBMs
    rbm_list = []
    # Size of inputs is the number of inputs in the training set
    input_size = inpX.shape[1]

    # For each RBM we want to generate
    for i, size in enumerate(RBM_hidden_sizes):
        print('RBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(RBM(input_size, size))
        input_size = size

    # For each RBM in our list
    for rbm in rbm_list:
        print('New RBM:')
        # Train a new one
        rbm.train(inpX)
        # Return the output layer
        inpX = rbm.rbm_outpt(inpX)

    nNet = NN(RBM_hidden_sizes, trX, trY)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    nNet.train()
