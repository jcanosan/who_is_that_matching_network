# !/usr/bin/env python

# recognise.py
# This module handles the image inputs from the robot camera, runs the model in
# matchingnets.py when it is necessary, receives the user inputs from
# dialogue.py and sends the system inputs to continue with the interaction.


import os
from glob import glob
import pickle

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from keras.applications.vgg16 import preprocess_input

from matchingnets import MatchingNets, collect_imgs_support_test, build_img_labels


def load_file(filepath):
    """
    Loads a pickle file.

    :param filepath: the path of the file to load.
    """
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data


class ImageRecogniser:
    def __init__(self):
        """Collects the specified images from the specified dataset"""

        # The dataset may be changed to another, but it need to match with the
        # one in dialogue.py when running both scripts
        self.dataset = "sota_dataset"

        # Collect the paths of the images to be used in this section
        # The number of shots may be changed to have more images per category
        # The number of categories (n_labels) may also be changed to limit the
        # number of categories that will be taken from the system
        # n_labels = None -> it will take all the categories
        png_support_set, _, self.all_labels = \
            collect_imgs_support_test(self.dataset, k_shot=5, n_labels=None)
        print(self.all_labels)

        # Build the parallel vectors of the images and the labels
        img_dataset_list_support, self.img_labels_support = \
            build_img_labels(png_support_set, self.all_labels, self.dataset)
        self.img_dataset_support = preprocess_input(img_dataset_list_support)

        # Initiate the Matching Networks model
        self.model = MatchingNets()

        # Encode the images
        self.img_dataset_support = \
            self.model.vgg16_encoding(self.img_dataset_support)

        # Declare the layers of the Matching Networks module
        self.model.model_layers(self.img_dataset_support,
                                self.img_labels_support)

        # Train the Matching Networks on the encoded images
        self.model.run_model(self.img_dataset_support, self.img_labels_support,
                             self.img_dataset_support, self.img_labels_support)

        # Set the user utterance to an empty string to avoid an error in the
        # ROS callback (image_callback)
        self.user_iter = ""

        # The following variables are only used when activating the depth
        # filtering
        self.current_depth = None
        self.region_of_interest = (1, 1000)

        # Declare variables that will be used later for image from the camera
        self.display_img = None
        self.test_img = None

        # Initiate rospy nodes to communicate with dialogue.py
        rospy.init_node("recognition_feedback")
        self.system_iter_pub = rospy.Publisher("/system_iter_topic", String,
                                               queue_size=1)
        self.user_iter_sub = rospy.Subscriber("/user_iter_topic", String,
                                              self.user_iter_callback,
                                              queue_size=1)
        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_color", Image,
                                        self.image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image,
                                          self.depth_callback, queue_size=1)

    def user_iter_callback(self, user_iter_topic):
        """
        Takes the inputs of the /user_iter_topic that come from dialogue.py.

        :param user_iter_topic: the user iteration.
        """
        self.user_iter = user_iter_topic.data

        # If the user input starts with "RTRAIN:", retrain the model on the new
        # image
        if self.user_iter.startswith("RTRAIN:"):
            label = self.user_iter.split(":")[1]
            self.save_img(label)
            self.retrain_model(label)

        # Elif the user input starts with "SAVE_IMG:", saves the last image
        # taken by the camera with its label
        elif self.user_iter.startswith("SAVE_IMG:"):
            label = self.user_iter.split(":")[1]
            self.save_img(label)
            print("Image saved!")
            self.system_iter_pub.publish("image_saved")
            self.user_iter = "unk"

        # Elif the user input begins by "NEW_LAB:", retrain the model on 5
        # images of the new label
        elif self.user_iter.startswith("NEW_LAB:"):
            label = self.user_iter.split(":")[1]
            self.save_img(label)
            self.learn_new_label(label)


    def image_callback(self, image_data):
        """
        Takes the inputs from the camera (topic /camera/rgb/image_color).
        Opens on screen a frame showing what the camera is seeing.

        If the user presses "q" on the frame, stops the program.

        If the user presses "r" on the frame or the self.user_iter is
        "what is this", then it will make a prediction of the image, print it
        on the console and publish it on the topic /system_iter_topic.

        :param image_data: the image input from /camera/rgb/image_color.
        """

        try:
            rgb_frame = np.array(CvBridge().imgmsg_to_cv2(image_data, "bgr8"),
                                 dtype=np.uint8)
        except CvBridgeError, e:
            print(e)

        # To show rgb + depth sensor filter
        # depth_frame = self.current_depth
        # self.display_img = self.process_image(rgb_frame, depth_frame)
        # cv2.imshow("Processed image", self.display_img)


        # To show only rgb image
        cv2.imshow("RGB image", rgb_frame)

        keystroke = chr(cv2.waitKey(1) & 0xFF).lower()
        self.display_img = cv2.resize(rgb_frame, (224, 224))

        if keystroke == 'q':
            rospy.signal_shutdown("The user hit q to exit.")

        elif keystroke == 'r' or self.user_iter == "RECOGNIZE":
            self.test_img = np.asarray(self.display_img, dtype=np.float32)
            self.test_img = np.expand_dims(self.test_img, axis=0)
            self.test_img = preprocess_input(self.test_img)
            self.test_img = self.model.vgg16_encoding(self.test_img)

            # Predict and decode labels
            dummy_label = np.array([[0.0]])
            predict = \
                self.model.predict_evaluate(self.img_dataset_support,
                                            self.img_labels_support,
                                            self.test_img, dummy_label)[0]
            labels_pred = [x for x in sorted(zip(predict[0], self.all_labels),
                                             reverse=True)]

            # Print scores for all the labels to compare
            print("\nPrediction:")
            for i in labels_pred:
                print(i)

            if self.user_iter == "RECOGNIZE":
                # Publish the predicted label
                self.system_iter_pub.publish("{0}:{1}".format(labels_pred[0][1],
                                                              labels_pred[0][0]))
                self.user_iter = "unk"

        elif self.user_iter.startswith("THIS_IS:"):
            self.test_img = np.asarray(self.display_img, dtype=np.float32)
            self.test_img = np.expand_dims(self.test_img, axis=0)
            self.test_img = preprocess_input(self.test_img)
            self.test_img = self.model.vgg16_encoding(self.test_img)

            self.system_iter_pub.publish("ANSWER")
            self.user_iter = "unk"
        else:
            pass

    def depth_callback(self, depth_data):
        """
        Takes the current image of the depth sensor and saves it in the variable
        "current_depth" to use in image_callback.

        :param depth_data: the image from the depth sensor.
        """
        try:
            # The depth image is a single-channel float32 image
            depth_frame = CvBridge().imgmsg_to_cv2(depth_data, "16UC1")
        except CvBridgeError, e:
            print e

        depth_array = np.array(depth_frame, dtype=np.float32)
        depth_array = np.roll(depth_array, -15)
        self.current_depth = np.copy(depth_array)

        # The following lines are just to test the depth filtering
        # max_depth = self.region_of_interest[1]
        # depth_array[depth_array < self.region_of_interest[0]] = max_depth
        # depth_array[depth_array > self.region_of_interest[1]] = max_depth
        # Normalize the depth image to fall between 0 (black) and 1 (white)
        # cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)

        # Display the result of the depth sensor
        # cv2.imshow("Depth image", depth_array)
        # keystroke = chr(cv2.waitKey(1) & 0xFF).lower()
        # if keystroke == 'q':
        #     rospy.signal_shutdown("The user hit q to exit.")

    def process_image(self, frame, depth_frame):
        """
        Applies depth filtering to the current RGB frame given a depth frame.

        :param frame:
        :param depth_frame:
        :return:
        """
        # set every pixel that is outside of the RoI to white (255,255,255).
        frame[np.tile(depth_frame > self.region_of_interest[1], (1, 1))] = 255
        frame[np.tile(depth_frame < self.region_of_interest[0], (1, 1))] = 255
        return frame

    def save_img(self, label):
        """
        Saves the image into the folder "extended_dataset".

        :param label: the label of the image.
        """
        dataset_to_save = self.dataset
        # New images will be saved outside SOTA dataset if the line below is
        # uncommented
        # dataset_to_save = "extra-dataset"

        label_path = "utils/datasets/{0}/{1}".format(dataset_to_save, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        img_num = 0
        while os.path.exists("{0}/{1}{2}.png".format(label_path, label, img_num)):
            img_num += 1

        img_path = "{0}/{1}{2}.png".format(label_path, label, img_num)

        cv2.imwrite(img_path, self.display_img)

    def retrain_model(self, label):
        """
        Retrains the model on the new input from the camera.

        :param label: the correct label of the image input.
        """
        print("List of all the labels:")
        print(self.all_labels)

        # Position of the new label
        new_img_label = self.all_labels.index(label)

        # Build vector of labels for the new image to train.
        img_labels = np.eye(len(self.all_labels))
        new_img_label = [img_labels[np.squeeze(new_img_label)]]

        self.img_dataset_support = \
            np.append(self.img_dataset_support, self.test_img, axis=0)

        self.img_labels_support = \
            np.append(self.img_labels_support, new_img_label, axis=0)
        print(self.img_dataset_support.shape)
        # print(self.img_labels_support)
        # print(self.img_labels_support.shape)

        self.model.model_layers(self.img_dataset_support,
                                self.img_labels_support)
        self.model.run_model(self.img_dataset_support, self.img_labels_support,
                             self.img_dataset_support, self.img_labels_support)

        print("Retrain complete!")
        self.system_iter_pub.publish("retrain_complete")
        self.user_iter = "unk"

    def learn_new_label(self, label):
        """
        Retrains the model on the 5 available images of the new label

        :param label:
        :return:
        """
        self.all_labels.append(label)
        print("New list of all the labels:")
        print(self.all_labels)

        # Add a new column to the labels vector of the support dataset
        # that represents the new label
        new_column = np.zeros((self.img_labels_support.shape[0], 1),
                              dtype=np.float32)
        self.img_labels_support = np.append(self.img_labels_support, new_column,
                                            axis=1)

        # Load the 5 images of the new label to add to the model
        img_paths = sorted([img for img in glob(
            "utils/datasets/{0}/{1}/*".format(self.dataset, label))])

        images_new_label, labels_new_label = \
            build_img_labels(img_paths, self.all_labels, self.dataset)

        images_new_label = preprocess_input(images_new_label)
        images_new_label = self.model.vgg16_encoding(images_new_label)

        # Append the vectors of the new images and label to the support set
        self.img_dataset_support = np.append(self.img_dataset_support,
                                             images_new_label, axis=0)

        self.img_labels_support = np.append(self.img_labels_support,
                                            labels_new_label, axis=0)
        # print(self.img_labels_support)
        # print(self.img_dataset_support.shape)
        # print(self.img_labels_support.shape)

        self.model.model_layers(self.img_dataset_support,
                                self.img_labels_support)
        self.model.run_model(self.img_dataset_support, self.img_labels_support,
                             self.img_dataset_support, self.img_labels_support)

        print("Retrain complete!")
        self.system_iter_pub.publish("retrain_complete")
        self.user_iter = "unk"


if __name__ == '__main__':
    # Run ROS-Kinect
    ImageRecogniser()
    rospy.spin()
