"""
File contains code for training DNN for cifar10 dataset classification
Trained model is saved in the specified model

Created on: July 17, 2018
Author: Harshal
"""

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    import cifar10_input
    import cifar10_model

    BATCH_SIZE = 128
    NO_OF_EPOCHS = 1000
    INITIAL_LEARNING_RATE = 1.0
    DECAY_STEP = 10000
    DECAY_RATE = 0.1
    LAMBDA = 0.001

    image = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
    label = tf.placeholder(tf.int32)

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, step, DECAY_STEP, DECAY_RATE, staircase=True)
    logits_train = cifar10_model.dnn(image_queue, training = True)
    tf.get_variable_scope().reuse_variables()
    logits_test = cifar10_model.dnn(image_queue, training = False)
    loss, train_step = cifar10_model.train(logits_train, label_queue, learning_rate, LAMBDA, step)
    accuracy = cifar10_model.old_evaluate(logits_test, label_queue)

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]

    session_args = {
        'checkpoint_dir': './trained_model',
        'save_checkpoint_steps': 300
    }
    with tf.train.MonitoredTrainingSession(**session_args) as sess:

        count = 1
        for i in range(5):
            cifar10_dataset = cifar10_input.unpickle(filename_list[i])
            image_in = np.reshape(cifar10_dataset[b'data'], (-1, 32, 32, 3))
            label_in = cifar10_dataset[b'labels']
            sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

            while True:
                try:
                    loss_value, _, accuracy_value = sess.run([loss, train_step, accuracy])
                    if count % 100 == 0:
                        print("Step: %6d,\tLoss: %8.4f,\tAccuracy: %0.4f" % (count, loss_value, accuracy_value))
                    count += 1
                except tf.errors.OutOfRangeError:
                    break

        cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
        image_in = np.reshape(cifar10_dataset[b'data'], (-1, 32, 32, 3))
        label_in = cifar10_dataset[b'labels']
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        accuracy_value = sess.run(accuracy)
        print("Accuracy: ", accuracy_value)
