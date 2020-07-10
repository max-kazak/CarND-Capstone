from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from glob import glob
import os
from keras.models import load_model

TL_CLASS = 10
CONFIDENCE_THRESH = 0.3

SAVED_MODEL_COCO = "coco_inference_graph.pb"
# SAVED_MODEL_LENET = "lenet.h5"  # Python3 version
SAVED_MODEL_LENET = "lenet27.h5" # Python2.7 version


class TLClassifier(object):
    def __init__(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))

        self._load_coco(os.path.join(root_dir, "models", SAVED_MODEL_COCO))
        self._load_lenet(os.path.join(root_dir, "models", SAVED_MODEL_LENET))

    def _load_lenet(self, path):
        self.class_model = load_model(path)
        self.class_graph = tf.get_default_graph()
        self.tlclasses = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN]
        self.tlclassesnames = ['Red', 'Yellow', 'Green']

    def _load_coco(self, path):
        # Use ssd_modilenet_v1_coco model from Tensorflow pre-trained models:
        # https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md
        # Part of the code is a courtesy of https://www.activestate.com/blog/using-pre-trained-models-tensorflow-go/
        self.dg = tf.Graph()
        with self.dg.as_default():
            gdef = tf.GraphDef()
            with open(path, 'rb') as f:
                gdef.ParseFromString(f.read())
                tf.import_graph_def(gdef, name="")

            self.session = tf.Session(graph=self.dg)
            self.image_tensor = self.dg.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.dg.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.dg.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.dg.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.dg.get_tensor_by_name('num_detections:0')

    def _localize_lights(self, image):
        """ Localizes bounding boxes for lights using pretrained TF model
            expects BGR8 image
        """

        with self.dg.as_default():
            # switch from BGR to RGB. Important otherwise detection won't work
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tf_image_input = np.expand_dims(image, axis=0)
            # run detection model
            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)

            # Find first detection of traffic light (class=10).
            idx = -1
            for i, cl in enumerate(detection_classes.tolist()):
                if cl == TL_CLASS:
                    idx = i
                    break

            if idx == -1:
                # no traffic lights detected
                return None
            elif detection_scores[idx] < CONFIDENCE_THRESH:
                # not confident enough
                return None
            else:
                img_h, img_w = image.shape[0:2]
                box = detection_boxes[idx]  # normalized box coordinates
                box = np.array([int(box[0] * img_h), int(box[1] * img_w),  # top-left corner coords in px
                                int(box[2] * img_h), int(box[3] * img_w)])  # bottom-right corner coords in px
                box_h, box_w = (box[2] - box[0], box[3] - box[1])
                if (box_h < 20) or (box_w < 20):
                    # box too small
                    return None
                elif (box_h / box_w < 1.6):
                    # wrong ratio
                    return None
                else:
                    # all good
                    rospy.loginfo('tl found on the image; box({}x{}): {} conf: {}'.format(box_h, box_w,
                                                                                          box, detection_scores[idx]))
                    return box

    def _classify_state(self, image):
        """
            Classify state of the  traffic light on the image.

            Args:
                image: 32x32x3 image in BGR format

            Returns:
                state: traffic light state
        """
        state = TrafficLight.UNKNOWN
        img_resize = np.expand_dims(image, axis=0).astype('float32')
        with self.class_graph.as_default():
            predict = self.class_model.predict(img_resize)[0]
            most_prob = np.argmax(predict)
            state = self.tlclasses[most_prob]
            state_name = self.tlclassesnames[most_prob]

            rospy.loginfo("light state confirmed to be {}({}) with confidence {}".format(state_name, state, predict[most_prob]))

        return state

    def get_classification(self, image):
        """
            Determines the color of the traffic light in the image

            Args:
                image (cv::Mat): image containing the traffic light

            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        box = self._localize_lights(image)
        if box is None:
            return TrafficLight.UNKNOWN

        tl_img = image[box[0]:box[2], box[1]:box[3]]
        tl_img = cv2.resize(tl_img, (32, 64))

        return self._classify_state(tl_img)

    def get_tl_image(self, image):
        """
            Get cutout of the traffic light. Used to gather training data.

            Args:
                image: camera image in BGR format

            Return:
                tl_image: BGR 32x32px image of the traffic light
        """
        box = self._localize_lights(image)
        if box is None:
            return None

        tl_img = image[box[0]:box[2], box[1]:box[3]]
        rospy.loginfo("found tl box ({}x{})".format(tl_img.shape[0], tl_img.shape[1]))

        return cv2.resize(tl_img, (32, 64))
