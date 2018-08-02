"""
File contains code for inferring input image with trained model for cifar10 dataset classification

Created on: July 18, 2018
Author: Harshal
"""

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    import cifar10_input
    import cifar10_model

    image = tf.placeholder(tf.float32, shape = (None, 32, 32, 3))
    label = tf.placeholder(tf.int32)

    dataset_iterator = cifar10_input.input_dataset(image, label, 256, 1)
    data = dataset_iterator.get_next()
    image_queue = data["features"]
    label_queue = data["label"]

    with tf.device('/cpu:0'):
        logits = cifar10_model.dnn(image_queue, training = False)
        prediction, probability = cifar10_model.predict(logits)
        _, accuracy = cifar10_model.evaluate(logits, label_queue)

    path = './dataset/cifar-10-batches-py'

    saver_handle = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        saver_handle.restore(sess, "./trained_model/model.ckpt")
        cifar10_dataset = cifar10_input.unpickle(path + '/test_batch')
        image_in = np.reshape(cifar10_dataset[b'data'], (-1, 32, 32, 3))
        label_in = cifar10_dataset[b'labels']
        sess.run(dataset_iterator.initializer, feed_dict = {image: image_in, label: label_in})

        prediction_out, probability_out, actual = sess.run([prediction, probability, label_queue])
        #print("Prediction: %d with Probability: %f\nActual: %d" % (prediction_out, probability_out, actual))
        print("Accuracy: %f" % (accuracy.eval()))
