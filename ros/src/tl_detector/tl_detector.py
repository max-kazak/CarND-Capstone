#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np
import PyKDL

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

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

        rospy.loginfo('Traffic light detector initialized')

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
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
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose, waypoints=None, mode='omni'):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
            waypoints: list of waypoints to search (default=self.waypoints)
            mode: 'omni' - all directions, 'forward' - forward direction

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if waypoints is None:
            waypoints = self.waypoints

        # search using nearest distance
        min_dist = float('inf')
        min_idx = -1
        search_range = 300  # don't bother if traffic light is too far

        for wp_idx, wp in enumerate(waypoints):
            dist = calc_pose_dist(wp.pose.pose, pose)

            if (dist < search_range) and (dist < min_dist) :
                if mode == 'omni':
                    min_dist = dist
                    min_idx = wp_idx

                elif mode == 'forward':
                    po = pose.orientation
                    wpo = wp.pose.pose.orientation
                    wpp = wp.pose.pose.position

                    car_vector = PyKDL.Rotation.Quaternion(po.x, po.y, po.z, po.w) * PyKDL.Vector(1, 0, 0)  # change the reference frame of 1,0,0 to the orientation of the car
                    wp_vector = PyKDL.Vector(wpp.x - pose.position.x, wpp.y - pose.position.y, 0) # shift wp vector to car origin

                    # dot product is the cosinus of angle between both
                    angle = np.arccos(PyKDL.dot(car_vector, wp_vector) / car_vector.Norm() / wp_vector.Norm())

                    if angle < np.pi / 2:
                        # within 90deg view angle
                        min_dist = dist
                        min_idx = wp_idx
                else:
                    rospy.logerr('Unsupported search method {}'.format(mode))
                    return min_idx

        return min_idx



    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        # check inputs
        if self.pose is None or self.waypoints is None or self.light_classifier is None:
            return -1, TrafficLight.UNKNOWN

        # START: Find upcoming light
        light_idx = self.get_closest_waypoint(self.pose.pose, self.lights, "forward")

        if light_idx == -1:
            # no light ahead
            return -1, TrafficLight.UNKNOWN
        # END: Find upcoming light

        # START: waypoint closes to the upcoming stop line for a traffic light
        stop_wp_idx = self.get_closest_waypoint(stop_line_positions[light_idx].pose.pose,
                                                      self.waypoints.waypoints,
                                                      mode='omni')
        if stop_wp_idx == -1:
            rospy.logerr('No waypoint found close to traffic light')
            return -1, TrafficLight.UNKNOWN
        # END: waypoint closes to the upcoming stop line for a traffic light

        # START: get traffic light state
        state = self.lights[light_idx].state  # CHEAT MODE
        # state = self.get_light_state(self.lights[light_idx]) # use classifier
        # END: get traffic light state

        return stop_wp_idx, state


def calc_pose_dist(self, a, b):
    return math.sqrt((a.position.x - b.position.x) ** 2 + (a.position.y - b.position.y) ** 2)

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
