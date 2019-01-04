from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np


def show_image(mnist):
    images = mnist.train.images
    labels = mnist.train.labels

    fig, axes = plt.subplots(2, 5)

    axes = axes.flatten()

    for i in range(10):
        image = np.asarray(images[i + 10], dtype=np.float32).reshape([28, 28])
        print(np.argmax(np.asarray(labels[i + 10]), axis=0))
        axes[i].imshow(image, cmap='Greys', interpolation='nearest')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    show_image(mnist)
