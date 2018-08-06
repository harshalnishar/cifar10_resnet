"""
File contains ResNet model for cifar10 dataset classification

Created on: July 13, 2018
Author: Harshal
"""

import tensorflow as tf
import resnet_model
import numpy as np

def dnn(image, training):
    """
    function which calls resnet model with proper initialization for cifar10 model
    :param image: input image tensor
    :return: model output tensor node
    """

    resnet_object = resnet_model.Model(resnet_size = 32,
                                       bottleneck = False,
                                       num_classes = 10,
                                       num_filters = 16,
                                       kernel_size = 3,
                                       conv_stride = 1,
                                       first_pool_size = None,
                                       first_pool_stride = None,
                                       block_sizes = [5, 5, 5],
                                       block_strides = [1, 2, 2],
                                       final_size = 64,
                                       data_format = 'channels_last'
                                       )
    logits = resnet_object(image, training = training)
    return logits

def predict(logits):
    """
    function outputs the predicted class based on input logits
    :param logits: logits tensor
    :return: returns predicted output with prediction probability
    """
    prediction = tf.cast(tf.argmax(logits, axis = 1), tf.int32)
    probability = tf.reduce_max(tf.nn.softmax(logits), axis = 1)
    return prediction, probability

def old_evaluate(logits, labels):
    prediction, _ = predict(logits)
    match = tf.equal(labels, prediction)
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))
    return accuracy

def evaluate(logits, labels):
    """
    function to evaluate the logits output against labels
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :return: prediction accuracy
    """
    prediction, _ = predict(logits)
    accuracy, accuracy_op = tf.metrics.accuracy(labels = labels, predictions = prediction)
    return accuracy, accuracy_op

def train(logits, labels, learning_rate):
    """
    function to train the dnn model for cifar10 training set
    :param logits: logits tensor
    :param labels: normal (not one hot) labels tensor
    :param learning_rate: initial learning rate
    :return: trainig operation
    """
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    optimizer_step = optimizer.minimize(loss)
    return loss, optimizer_step


if __name__ == "__main__":
    import cifar10_input

    BATCH_SIZE = 128
    NO_OF_EPOCHS = 100
    LEARNING_RATE = 1

    image = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
    label = tf.placeholder(tf.int32)
    training_mode = tf.placeholder(tf.bool, shape = ())

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    data = dataset_iterator.get_next()

    image_queue = data["features"]
    label_queue = data["label"]

    logits = dnn(image_queue, training = training_mode)
    loss, train_step = train(logits, label_queue, LEARNING_RATE)
    accuracy = old_evaluate(logits, label_queue)

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        cifar10_dataset = cifar10_input.unpickle(filename_list[0])
        image_in = np.reshape(cifar10_dataset[b'data'], (-1, 32, 32, 3))
        label_in = cifar10_dataset[b'labels']
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        count = 1
        while True:
            try:
                loss_value, _, accuracy_value = sess.run([loss, train_step, accuracy],
                                                         feed_dict = {training_mode: True})
                if count % 100 == 0:
                    print("Step: %6d,\tLoss: %8.4f,\tAccuracy: %0.4f" % (count, loss_value, accuracy_value))
                count += 1
            except tf.errors.OutOfRangeError:
                break

        cifar10_dataset = cifar10_input.unpickle(filename_list[1])
        image_in = np.reshape(cifar10_dataset[b'data'], (-1, 32, 32, 3))
        label_in = cifar10_dataset[b'labels']

        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})
        accuracy_value = sess.run(accuracy, feed_dict = {training_mode: True})
        print("Accuracy: ", accuracy_value)
