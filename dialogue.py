#!/usr/bin/env python
import time
import sys
from glob import glob
import rospy
from std_msgs.msg import String


class Dialogue():
    def __init__(self):
        rospy.init_node("user_iter")
        self.user_iter_pub = rospy.Publisher("user_iter_topic", String,
                                             queue_size=1)
        self.system_iter_sub = rospy.Subscriber("/system_iter_topic", String,
                                                self.system_iter_callback,
                                                queue_size=1)

        self.system_iter = None
        self.label_score = None
        self.dataset = "sota_dataset"

        # Greeting
        print("S> Hello!")

        while True:
            self.user_input = ""
            self.user_input_yn = ""
            self.state_init()

    def state_init(self):
        whats_this = ("what is this", "what's this", "w")
        this_is = ("this is a", "this is an")
        accepted_utt = whats_this + this_is

        while not self.user_input.startswith(accepted_utt):
            time.sleep(1)
            self.user_input = raw_input("\n\nS> What do you want me to learn?"
                                        "\n\nU> ").lower()
            # If the user writes "exit", stop the run
            if self.user_input == "exit":
                rospy.on_shutdown(self.shutdown_callback)
                sys.exit()
            # If the user writes any other thing
            elif not self.user_input.startswith(accepted_utt):
                print("\nS> Sorry, I didn't understand you.")

        if self.user_input in whats_this:
            self.state_whats_this()
        elif self.user_input.startswith(this_is):
            if self.user_input.startswith("this is an"):
                label = self.user_input.split("this is an ")[1].lower()
            else:
                label = self.user_input.split("this is a ")[1].lower()
            label = label.replace(" ", "_")
            self.state_this_is(label)

    def state_this_is(self, label):
        self.user_iter_pub.publish("THIS_IS:{0}".format(label))

        # For 10 seconds, checks every 500ms for an answer from the system.
        for i in range(21):
            if self.system_iter is None and i < 20:
                time.sleep(0.5)
            elif self.system_iter is None and i == 20:
                print("\nS> There has been some problem when trying to get "
                      "feedback from the camera.")
                print("\nS> Please, check that the camera is connected "
                      "and recognise.py is running.")
                sys.exit()
            else:
                break
        self.system_iter = None
        self.state_this_is_train_or_new_label(label)

    def state_this_is_train_or_new_label(self, label):
        """For the 'this is...' action, check the number of images of the
        specified label and act according to it."""
        num_images = self.check_num_images_label(label)
        if num_images == "-5":
            print("\nS> Please, show me more examples of {0}.".format(label))
            self.save_img(label)
        elif num_images == "=5":
            print("\nS> I am learning {0}.".format(label))
            self.train_new_label(label)
        else:
            print("\nS> I am updating my systems on {0}.".format(label))
            self.train_model(label)

    def state_whats_this(self):
        self.user_iter_pub.publish("RECOGNIZE")

        # For 10 seconds, checks every 500ms for an answer from the system.
        for i in range(21):
            if self.system_iter is None and i < 20:
                time.sleep(0.5)
            elif self.system_iter is None and i == 20:
                print("\nS> There has been some problem when trying to get "
                      "feedback from the camera.")
                print("\nS> Please, check that the camera is connected "
                      "and recognise.py is running.")
                sys.exit()
            else:
                break

        # Split the <object guess> from the confidence score
        self.label_score = self.system_iter.split(":")

        # Check the confidence score and choose output according to that
        if float(self.label_score[1]) >= 0.8:
            self.state_score80()
        elif float(self.label_score[1]) >= 0.5:
            self.state_score50()
        elif float(self.label_score[1]) >= 0.3:
            self.state_score30()
        else:
            self.state_score00()

    def state_score80(self):
        if self.system_iter.startswith(("a", "e", "i", "o", "u")):
            print("\nS> This is an {0}. Am I right?".
                  format(self.label_score[0]))
        elif not self.system_iter.startswith(("a", "e", "i", "o", "u")):
            print("\nS> This is a {0}. Am I right?".
                  format(self.label_score[0]))
        self.system_iter = None
        self.state_answer_yes_no()
        self.state_80_50_train()

    def state_score50(self):
        if self.system_iter.startswith(("a", "e", "i", "o", "u")):
            print("\nS> I think this is an {0}. Am I right?".
                  format(self.label_score[0]))
        elif not self.system_iter.startswith(("a", "e", "i", "o", "u")):
            print("\nS> I think this is a {0}. Am I right?".
                  format(self.label_score[0]))
        self.system_iter = None
        self.state_answer_yes_no()
        self.state_80_50_train()

    def state_score30(self):
        if self.system_iter.startswith(("a", "e", "i", "o", "u")):
            print("\nS> I am not sure. Is this an {0}?".
                  format(self.label_score[0]))
        elif not self.system_iter.startswith(("a", "e", "i", "o", "u")):
            print("\nS> I am not sure. Is this a {0}?".
                  format(self.label_score[0]))
        self.system_iter = None
        self.state_answer_yes_no()
        self.state_30_train()

    def state_score00(self):
        self.system_iter = None
        self.state_00_train()

    def state_answer_yes_no(self):
        """ User has to answer 'yes' or 'no'."""
        while self.user_input_yn not in ("yes", "no"):
            self.user_input_yn = raw_input("\nU> ").lower()
            # If the user writes "exit", stop the run
            if self.user_input_yn == "exit":
                rospy.on_shutdown(self.shutdown_callback)
                sys.exit()
            # If the user writes any other thing
            elif self.user_input_yn not in ("yes", "no"):
                print('\nS> Please, answer "yes" or "no".')
                print('\nS> You can also write "exit" to close the program.')

    def state_80_50_train(self):
        if self.user_input_yn == "yes":
            print("\nS> Great!")

        elif self.user_input_yn == "no":
            label = raw_input("\nS> What is this, then? Write only the name of "
                              "the object.\n\nU> ").lower()
            # If the user writes "exit", stop the run
            if label == "exit":
                rospy.on_shutdown(self.shutdown_callback)
                sys.exit()
            else:
                self.state_whats_this_train_or_new_label(label)

    def state_30_train(self):
        if self.user_input_yn == "yes":
            self.state_whats_this_train_or_new_label(self.label_score[0])

        elif self.user_input_yn == "no":
            label = raw_input("\nS> What is this, then? Write only the name of "
                              "the object.\n\nU> ").lower()
            # If the user writes "exit", stop the run
            if label == "exit":
                rospy.on_shutdown(self.shutdown_callback)
                sys.exit()
            else:
                self.state_whats_this_train_or_new_label(label)

    def state_00_train(self):
        print("\nS> I don't know what this is. Please, tell me. Write only the "
              "name of the object.")
        label = raw_input("\nU> ").lower()
        # If the user writes "exit", stop the run
        if label == "exit":
            rospy.on_shutdown(self.shutdown_callback)
            sys.exit()
        else:
            self.state_whats_this_train_or_new_label(label)

    def state_whats_this_train_or_new_label(self, label):
        """For the 'what's this' query, check the number of images of the
        specified label and act according to it."""
        num_images = self.check_num_images_label(label)
        if num_images == "-5":
            print("\nS> I didn't know about {0}. Please, show me more examples "
                  "of it.".format(label))
            self.save_img(label)
        elif num_images == "=5":
            print("\nS> I didn't know about {0}, but I am learning right now."
                  .format(label))
            self.train_new_label(label)
        else:
            print("\nS> I am updating my systems on {0}.".format(label))
            self.train_model(label)

    def check_num_images_label(self, label):
        """Checks the number of images that the label has."""
        label_path = "utils/datasets/{0}/{1}".format(self.dataset, label)
        # TODO: CHECK BOTTOM LINE RUNS
        label_path_extra = "utils/datasets/{0}/{1}".format("extra-dataset", label)
        label_num_images = len(glob("{0}/*".format(label_path))) + \
                           len(glob("{0}/*".format(label_path_extra)))

        if label_num_images < 4:
            return("-5")
        elif label_num_images == 4:
            return("=5")
        else:
            return("+5")

    def train_new_label(self, label):
        """
        Sends to recognise.py the information to retrain the model on a new
        label and waits until it has finished.

        :param label: the label to learn.
        """
        self.user_input = "NEW_LAB:{0}".format(label)
        self.user_iter_pub.publish(self.user_input.replace(" ", "_"))
        print("\nS> It will only take me a few seconds...")
        while self.system_iter != "retrain_complete":
            time.sleep(1)
        print("\nS> Update complete!")
        self.system_iter = None

    def train_model(self, label):
        """
        Sends to recognise.py the information to retrain the model on the new
        image and waits until it has finished.

        :param label: the label of the image.
        """
        self.user_input = "RTRAIN:{0}".format(label)
        self.user_iter_pub.publish(self.user_input.replace(" ", "_"))
        print("\nS> It will only take me a few seconds...")
        while self.system_iter != "retrain_complete":
            time.sleep(1)
        print("\nS> Update complete!")
        self.system_iter = None

    def save_img(self, label):
        """
        Sends to recognise.py the information to save the last image taken on
        the right label.
        :param label: the label of the image.
        """
        self.user_input = "SAVE_IMG:{0}".format(label)
        self.user_iter_pub.publish(self.user_input.replace(" ", "_"))
        while self.system_iter != "image_saved":
            time.sleep(1)
        self.system_iter = None

    def system_iter_callback(self, system_iter_topic):
        self.system_iter = system_iter_topic.data

    def shutdown_callback(self):
        print("\nS> Bye, bye!")


if __name__ == '__main__':
    Dialogue()
    rospy.spin()
