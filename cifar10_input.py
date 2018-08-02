"""
File contains routine for importing and manipulating input data

Created on: July 6, 2018
Author: Harshal
"""

import tensorflow as tf

def unpickle(file):
    """
    function to read pickle file and return a dictionary of dataset
    :param file: pickle file name with full path from current directory
    :return dataset: dictionary of dataset imported from pickle file
    """
    import pickle
    with open(file, 'rb') as file_object:
        dataset = pickle.load(file_object, encoding='bytes')
    return dataset

def input_dataset(image, label, batch_size, epochs):
    """
    function to create and return dataset iterator
    :param image: input image tensor
    :param label: input label tensor
    :param batch_size: batch size for dataset batching
    :param epochs: number of epochs (dataset repetition)
    :return dataset_iterator: initialisable iterator over dataset
    """

    dataset_object = tf.data.Dataset.from_tensor_slices({"features": image, "label": label})
    dataset_object = dataset_object.repeat(epochs)
    dataset_object = dataset_object.shuffle(1000)
    dataset_object = dataset_object.batch(batch_size)
    dataset_iterator = dataset_object.make_initializable_iterator()

    return dataset_iterator


if __name__ == '__main__':

    BATCH_SIZE = 128
    NO_OF_EPOCHS = 10

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]
    cifar10_dataset = unpickle(filename_list[0])
    image_in = cifar10_dataset[b'data']
    label_in = cifar10_dataset[b'labels']

    image = tf.placeholder(tf.uint8)
    label = tf.placeholder(tf.uint8)

    dataset_iterator = input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    with tf.Session() as sess:
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})
        data = sess.run(dataset_iterator.get_next())
        print(data)
