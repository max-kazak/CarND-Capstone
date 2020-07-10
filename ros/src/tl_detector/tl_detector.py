#!/usr/bin/env python
import os
import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3
GATHER_DATA = False
USE_EVERY_N_IMG = 3

root_dir = os.path.dirname(os.path.realpath(__file__))


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Data gathering counters
        self.tl_cnt = 0  # traffic light counter
        self.last_saved_state = TrafficLight.UNKNOWN

        # control what images to process
        self.imgs_cnt = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.loginfo('Traffic light detector initialized')

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.waypoint_tree = KDTree([[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                     for waypoint in waypoints.waypoints])

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.imgs_cnt += 1
        if self.imgs_cnt % USE_EVERY_N_IMG > 0:
            return
        self.imgs_cnt = 0

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            rospy.loginfo('New TL state accepted: {}'.format(state))
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            rospy.loginfo('Publish new stop wp: {}'.format(light_wp))
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            rospy.loginfo('Publish last stop wp: {}'.format(self.last_wp))
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([pose.position.x, pose.position.y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return None

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        if GATHER_DATA:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            tl_img = self.light_classifier.get_tl_image(cv_image)
            if (tl_img is not None) and (self.last_saved_state != light.state):
                cv2.imwrite(os.path.join(root_dir,
                                         'light_classification',
                                         'data',
                                         'tl{}_{}.png'.format(self.tl_cnt, light.state)),
                            tl_img)
                self.tl_cnt += 1
                self.last_saved_state = light.state

        # doneTODO: comment out when classificator is done
        # return light.state

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = -1

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        # check inputs
        if self.pose is None or self.waypoints is None or self.light_classifier is None:
            # probably not initialized yet
            return -1, TrafficLight.UNKNOWN

        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose)

            # Find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                stop_line_pos = Pose()
                stop_line_pos.position.x = stop_line_positions[i][0]
                stop_line_pos.position.y = stop_line_positions[i][1]
                wp_idx = self.get_closest_waypoint(stop_line_pos)
                # Find closest stop line waypoint index
                d = wp_idx - car_wp_idx
                if 0 <= d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = wp_idx

        if closest_light:
            # closest light found
            state = self.get_light_state(closest_light) # UNKNOWN=4, GREEN=2, YELLOW=1, RED=0
            # rospy.loginfo("Closest light: {}".format(closest_light))
            rospy.loginfo("state: {} line_wp_idx: {}".format(state, line_wp_idx))
            if state is not None:
                # was able to determine state of the light
                return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
