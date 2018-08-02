"""
File contains code for retraining of already trained model for cifar10 dataset classification

Created on: July 17, 2018
Author: Harshal
"""

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    import cifar10_input
    import cifar10_model

    BATCH_SIZE = 128
    NO_OF_EPOCHS = 100
    LEARNING_RATE = 0.1

    image = tf.placeholder(tf.float32, shape = (None, 32, 32, 3))
    label = tf.placeholder(tf.int32)

    dataset_iterator = cifar10_input.input_dataset(image, label, BATCH_SIZE, NO_OF_EPOCHS)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    logits_training = cifar10_model.dnn(image_queue, training = True)
    tf.get_variable_scope().reuse_variables()
    logits = cifar10_model.dnn(image_queue, training = False)
    loss, train_step = cifar10_model.train(logits_training, label_queue, LEARNING_RATE)
    accuracy = cifar10_model.old_evaluate(logits_training, label_queue)
    accuracy_final = cifar10_model.evaluate(logits_training, label_queue)

    path = './dataset/cifar-10-batches-py'
    filename_list = [(path + '/data_batch_%d' % i) for i in range(1, 6)]

    saver_handle = tf.train.Saver()

    with tf.Session() as sess:
        saver_handle.restore(sess, "./trained_model/model.ckpt")
        sess.run(tf.local_variables_initializer())

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

        saver_handle.save(sess, "./trained_model/model.ckpt")

        cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
        image_in = np.reshape(cifar10_dataset[b'data'], (-1, 32, 32, 3))
        label_in = cifar10_dataset[b'labels']
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        sess.run(tf.local_variables_initializer())
        _, accuracy_value = sess.run(accuracy_final)
        print("Accuracy: ", accuracy_value)
