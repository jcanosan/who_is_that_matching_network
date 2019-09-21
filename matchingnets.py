# matchingnets.py
# This module contains the neural network model (inside the object MatchingNets)
# and some functions for collecting the images needed ffor running a session
# with the robot.


from glob import glob
import pathlib
import time
import random

import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Input
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16


def collect_imgs_support_test(dataset, k_shot=5, exclude_test_img_per_label=0,
                              n_labels=None, file_extension=".png",
                              random_imgs=False, random_labs=False):
    """Collect the paths of the images for the support and the test sets and
    the labels.

    :param dataset: name of the dataset in the folder utils/datasets/ .
    :param k_shot: number of images to take per label for support dataset.
    :param exclude_test_img_per_label: images to exclude from the test split.
    :param n_labels: number of labels.
    :param file_extension: extension of the image files in the dataset.
    :param random_imgs: if True, the images are randomized.
    :param random_labs: if True, the labels are randomized. Only performed if
    n_labels is not None.

    :return: List of images for support and test and the list of labels."""

    dataset_path = "utils/datasets/{0}/".format(dataset)

    # Collect labels into a sorted list
    # If the dataset is Omniglot, each category is two nested folders
    # (e.g. "Angelic/character05")
    if dataset.startswith("omniglot"):
        all_png_files = sorted(glob("{0}*/*/*.png".format(dataset_path)))
        dataset_labels = \
            sorted(list(set("{0}/{1}".format(pathlib.Path(folder).parts[-3],
                                             pathlib.Path(folder).parts[-2])
                            for folder in all_png_files)))

        # If some label has less images than k_shot, remove it
        for label in dataset_labels:
            num_imgs_label = \
                len(sorted(glob("{0}{1}/*{2}".format(dataset_path, label,
                                                     file_extension))))
            if num_imgs_label < k_shot:
                dataset_labels.pop(dataset_labels.index(label))

        if n_labels is None:
            all_labels = dataset_labels
        else:
            if random_labs is True:
                all_labels = list(
                    set("{0}/{1}".format(pathlib.Path(folder).parts[-3],
                                         pathlib.Path(folder).parts[-2])
                        for folder in all_png_files))
                # Use a fixed random seed for reproducibility
                random.seed(32)
                random.shuffle(all_labels)
                all_labels = all_labels[:n_labels]
            else:
                all_labels = sorted(list(
                    set("{0}/{1}".format(pathlib.Path(folder).parts[-3],
                                         pathlib.Path(folder).parts[-2])
                        for folder in all_png_files)))[:n_labels]

    # If the dataset is NOT Omniglot
    else:
        all_png_files = sorted(glob("{0}*/*{1}".format(dataset_path, file_extension)))
        dataset_labels = sorted(list(set(pathlib.Path(folder).parts[-2] for
                                     folder in all_png_files)))

        # If some label has less images than k_shot, remove it
        for label in dataset_labels:
            num_imgs_label = \
                len(sorted(glob("{0}{1}/*{2}".format(dataset_path, label,
                                                     file_extension))))
            if num_imgs_label < k_shot:
                dataset_labels.pop(dataset_labels.index(label))

        if n_labels is None:
            all_labels = dataset_labels
        else:
            all_labels = dataset_labels[:n_labels]

    # Collect a list of images paths
    png_support_set = []
    png_test_set = []

    # Randomize the images
    if random_imgs is True:
        random_all_png_files = []
        for label in all_labels:
            label_png_path = []
            for png in sorted(glob("{0}{1}/*{2}".format(dataset_path, label,
                                                        file_extension))):
                label_png_path.append(png)
            # Use a fixed random seed for reproducibility
            random.seed(32)
            random.shuffle(label_png_path)
            random_all_png_files.append(label_png_path)

        for label_png in random_all_png_files:
            for png in label_png[:k_shot]:
                png_support_set.append(png)
            for png in label_png[exclude_test_img_per_label:]:
                png_test_set.append(png)

    # No randomize the images (take the first k_shot of each folder)
    elif random_imgs is False:
        for label in all_labels:
            for img in sorted(glob("{0}{1}/*{2}".format(dataset_path, label,
                                                        file_extension)))[:k_shot]:
                png_support_set.append(img)
            for img2 in sorted(glob("{0}{1}/*{2}".format(dataset_path, label,
                                                         file_extension)))[exclude_test_img_per_label:]:
                png_test_set.append(img2)


    return png_support_set, png_test_set, all_labels


def build_img_labels(all_png, all_labels, dataset):
    """
    Builds a list of images and a list with the indexes of the labels of the
    images.

    :param all_png: list with the paths of all the images.
    :param all_labels: list with all the labels.
    :param dataset: the dataset to read the images from. If it is Omniglot, the
    folder structure is different.

    :return: a list with the vectors of all the images and a vector with the
    labels of each category.
    """
    all_labels_index = {}
    for i, label in zip(range(len(all_labels)), all_labels):
        all_labels_index[label] = i

    img_dataset = []
    img_labels = []

    if dataset.startswith("omniglot"):
        for img in all_png:
            # load the images, transform into array and append into a list
            img_dataset.append(img_to_array(cv2.resize(cv2.imread(img, 0), (28, 28))))
            # build parallel list of labels for each image
            img_labels.append([all_labels_index["{0}/{1}".format(pathlib.Path(img).parts[-3], pathlib.Path(img).parts[-2])]])

    else:
        for img in all_png:
            # load the images, transform into array and append into a list
            img_dataset.append(img_to_array(cv2.imread(img)))
            # build parallel list of labels for each image
            img_labels.append([all_labels_index[pathlib.Path(img).parts[-2]]])

    img_dataset = np.array(img_dataset)
    # print(img_dataset)
    img_labels_index = np.eye(len(all_labels))
    img_labels = img_labels_index[np.squeeze(img_labels)]

    return img_dataset, img_labels


class MatchingNets:
    def __init__(self):

        self.vgg_layers = VGG16(include_top=False, weights='imagenet',
                                input_tensor=Input([224, 224, 3]))

        self.sess = None
        self.img_input_set_memory = None
        self.labels_input_set_memory = None
        self.img_input = None
        self.img_input_label = None
        self.preds = None
        self.optimizer_step = None
        self.loss = None
        self.accuracy = None

    def vgg16_encoding(self, images):
        """
        Encodes a set of images with VGG16 convolutional layers pretrained on
        ImageNet.

        :param images: the set of images.
        :return: the images encoded.
        """
        start_encode_time = time.time()
        encoded_images = self.vgg_layers.predict(images)
        end_encode_time = time.time()
        print('encoding time: {0:.2f}'.format(end_encode_time - start_encode_time))
        return encoded_images

    def cosine_similarities(self, target, support_set, num_imgs_support):
        """the c() function that calculate the cosine similarity between
        (embedded) support set and (embedded) target
        
        note: the author uses one-sided cosine similarity as zergylord
        said in his repo (zergylord/oneshot)

        Function adapted from:
        https://github.com/markdtw/matching-networks/blob/master/model.py
        """
        sup_similarity = []
        for img_support in support_set:
            similarity = tf.matmul(target, tf.expand_dims(img_support, 1))
            sup_similarity.append(similarity)

        return tf.reshape(tf.stack(sup_similarity, axis=1),
                          [-1, num_imgs_support])

    def model_layers(self, img_support, labels_support):
        """
        Declares the matching network model layers.

        :param img_support: images of the support set.
        :param labels_support: labels of the support set.
        :return: the model trained.
        """

        # Declare shapes of the datasets to train
        num_imgs_support = img_support.shape[0]
        out_cnn_shape1 = img_support.shape[1]
        out_cnn_shape2 = img_support.shape[2]
        out_cnn_shape3 = img_support.shape[3]
        num_labels = labels_support.shape[1]

        # Remove the memory of tensorflow graphs created
        tf.reset_default_graph()

        # g - support
        self.img_input_set_memory = tf.placeholder(
            shape=img_support.shape, dtype=tf.float32)
        img_set_memory = \
            tf.nn.avg_pool(self.img_input_set_memory,
                           ksize=[1, out_cnn_shape1, out_cnn_shape2, 1],
                           padding='SAME',
                           strides=[1, out_cnn_shape1, out_cnn_shape2, 1])
        img_set_memory = tf.reshape(img_set_memory,
                                    [num_imgs_support, out_cnn_shape3])

        self.labels_input_set_memory = \
            tf.placeholder(shape=(num_imgs_support, num_labels),
                           dtype=tf.float32)

        # f - target
        self.img_input = \
            tf.placeholder(
                shape=(None, out_cnn_shape1, out_cnn_shape2, out_cnn_shape3),
                dtype=tf.float32)
        img_pool = \
            tf.nn.avg_pool(self.img_input,
                           ksize=[1, out_cnn_shape1, out_cnn_shape2, 1],
                           padding='SAME',
                           strides=[1, out_cnn_shape1, out_cnn_shape2, 1])
        img_reshape = tf.reshape(img_pool, [-1, out_cnn_shape3])

        self.img_input_label = tf.placeholder(shape=(None,), dtype=tf.int64)

        # g - support function
        g_dense_w = tf.get_variable(shape=[out_cnn_shape3, out_cnn_shape3],
                                    dtype=tf.float32, name="g_dense_w")
        g_dense_b = tf.get_variable(shape=[out_cnn_shape3],
                                    dtype=tf.float32, name="g_dense_b")

        g_embeddings = []
        for img_memory in tf.unstack(img_set_memory):
            img_memory = tf.expand_dims(img_memory, 0)
            # We can potentially convert this loop into LSTM
            g_embed = tf.nn.relu(
                tf.nn.xw_plus_b(x=img_memory, weights=g_dense_w,
                                biases=g_dense_b))
            g_embed = tf.nn.l2_normalize(g_embed, 1)
            g_embed = tf.reshape(g_embed, [out_cnn_shape3])
            g_embeddings.append(g_embed)

        # f - target function
        f_dense_w = tf.get_variable(shape=[out_cnn_shape3, out_cnn_shape3],
                                    dtype=tf.float32, name="f_dense_w")
        f_dense_b = tf.get_variable(shape=[out_cnn_shape3],
                                    dtype=tf.float32, name="f_dense_b")

        f_embeddings = tf.nn.relu(
            tf.nn.xw_plus_b(x=img_reshape, weights=f_dense_w, biases=f_dense_b))
        f_embeddings = tf.nn.l2_normalize(f_embeddings, 1)

        # Cosine similarities
        img_cosine_similarities = self.cosine_similarities(f_embeddings,
                                                           g_embeddings,
                                                           num_imgs_support)

        # Compute softmax so the prediction scores are normalized (0-1)
        self.preds = tf.nn.softmax(tf.reshape(tf.matmul(
            img_cosine_similarities, self.labels_input_set_memory),
            [-1, num_labels]))

        # Loss and optimizer
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.img_input_label, logits=self.preds)

        optimizer = tf.train.AdamOptimizer()
        self.optimizer_step = optimizer.minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.img_input_label, tf.argmax(self.preds, axis=1)),
            dtype=tf.float32))

    def run_model(self, img_support, labels_support, img_target, labels_target,
                  epochs=100):
        """ Runs the model to train.

        :param img_support: images of the support set.
        :param labels_support: labels of the support set.
        :param img_target: images of the target set.
        :param labels_target: labels of the target set.
        :param epochs: number of epochs of the training.
        :return:
        """

        start_train_time = time.time()

        # Run training session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        steps_per_epoch = 1
        for ep in range(epochs):
            # start training
            for step in range(steps_per_epoch):
                # support
                x_memory = img_support
                y_memory = labels_support
                # target
                x = img_target
                y = np.argmax(labels_target, 1)

                feed_dict = {self.img_input_set_memory: x_memory,
                             self.labels_input_set_memory: y_memory,
                             self.img_input: x,
                             self.img_input_label: y}

                preds_value, loss_value, accuracy_value, _ = \
                    self.sess.run([self.preds, self.loss, self.accuracy,
                                   self.optimizer_step], feed_dict=feed_dict)

                if step % 100 == 0:
                    print('ep: {0:3d}, step: {1:3d}, loss: {2:.6f}, '
                          'acc: {3:.3f}'.format(ep + 1, step, loss_value,
                                                accuracy_value))

        end_train_time = time.time()
        print('training time: {0:.2f}'.format(end_train_time - start_train_time))

    def predict_evaluate(self, img_support, labels_support, img_test, labels_test):
        """ Runs the model to evaluate and predict.

        :param img_support: images encoded through VGG16 to use as support.
        :param labels_support: labels of the support set.
        :param img_test: images encoded through VGG16 to use as target.
        :param labels_test: labels of the target set.
        :return: the predictions and loss and accuracy metrics.
        """

        img_labels_test = labels_test

        x_memory = img_support
        y_memory = labels_support
        x = img_test
        y = np.argmax(img_labels_test, 1)

        feed_dict = {self.img_input_set_memory: x_memory,
                     self.labels_input_set_memory: y_memory,
                     self.img_input: x,
                     self.img_input_label: y}

        preds_value, loss_value, accuracy_value = self.sess.run(
            [self.preds, self.loss, self.accuracy], feed_dict=feed_dict)

        return preds_value, loss_value, accuracy_value
