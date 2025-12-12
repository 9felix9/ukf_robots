import math
import numpy as np
import csv
import os
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from geometry_msgs.msg import Quaternion

from kalman_positioning.ukf import UKF
from kalman_positioning.landmark_manager import LandmarkManager


class PositioningNode(Node):
    """
    ROS2 Node for 2D robot localization:
        - Prediction via odometry
        - Update via landmark observations
        - Logs ground truth, estimated state, and noisy odometry to CSV
    """

    def __init__(self):
        super().__init__("kalman_positioning_node")

        self.get_logger().info("Initializing Kalman Positioning Node…")
        self.gt_pose = None
        self.noisy_pose = None


        # -----------------------------------------------------------
        # CSV Logging
        # -----------------------------------------------------------
        # ===========================================================
        # EXPERIMENT PARAMETERS from script file
        # ===========================================================

        self.process_noise_xy = float(os.getenv("PROC_NOISE_XY", 0.05))
        self.process_noise_theta = float(os.getenv("PROC_NOISE_TH", 0.05))
        self.measurement_noise_xy = float(os.getenv("MEAS_NOISE_XY", 0.05))
        self.scenario_name = os.getenv("SCENARIO", "baseline")
        self.task_type = os.getenv("TASK_TYPE", "B1")



        # ===========================================================
        # CSV PATH (must be provided by runner)
        # ===========================================================
        csv_path = os.getenv("LOG_CSV")
        if not csv_path:
            raise RuntimeError("LOG_CSV environment variable not set. Runner must pass it.")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.logging_duration = 30.0
        self.logging_start_time = self.get_clock().now().nanoseconds * 1e-9

        self.csv_writer.writerow([
            "time",
            "gt_x", "gt_y",
            "est_x", "est_y",
            "noisy_x", "noisy_y"
        ])

        self.get_logger().info(f"Logging UKF data to {csv_path}")


        # -----------------------------------------------------------
        # Load landmarks
        # -----------------------------------------------------------
        self.landmark_manager = LandmarkManager()
        if not self.landmark_manager.load_from_csv("src/kalman_positioning/landmarks.csv"):
            self.get_logger().error("Could not load landmarks!")

        # -----------------------------------------------------------
        # Initialize UKF
        # -----------------------------------------------------------
        self.ukf = UKF(
            process_noise_xy=self.process_noise_xy,
            process_noise_theta=self.process_noise_theta,
            measurement_noise_xy=self.measurement_noise_xy,
            num_landmarks=len(self.landmark_manager.get_all_landmarks())
        )

        for idx, (lm_id, (lx, ly)) in enumerate(self.landmark_manager.get_all_landmarks().items()):
            self.ukf.landmarks[idx] = [lm_id, lx, ly]

        # Odometry initialization state
        self.first_odom_received = False
        self.last_time = None
        self.last_state = None

        # -----------------------------------------------------------
        # ROS Subscribers
        # -----------------------------------------------------------
        self.create_subscription(Odometry, "/robot_noisy", self.odometry_callback, 10)
        self.create_subscription(PointCloud2, "/landmarks_observed", self.landmark_callback, 10)
        self.create_subscription(Odometry, "/robot_gt", self.gt_callback, 10)

        # Publisher for estimated odometry
        self.estimated_pub = self.create_publisher(Odometry, "/robot_estimated_odometry", 10)

        self.get_logger().info("Kalman Positioning Node initialized.")


    # ----------------------------------------------------------------------
    # Helper: Check if logging is active
    # ----------------------------------------------------------------------
    def logging_active(self):
        if self.logging_start_time is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        return (now - self.logging_start_time) <= self.logging_duration


    # ----------------------------------------------------------------------
    # Ground truth callback
    # ----------------------------------------------------------------------
    def gt_callback(self, msg):
        self.gt_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)


    # ----------------------------------------------------------------------
    # Odometry → UKF Prediction
    # ----------------------------------------------------------------------
    def odometry_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # First odometry message initializes the UKF
        if not self.first_odom_received:
            self.last_time = current_time
            self.last_state = np.array([x, y, theta])
            self.first_odom_received = True

            self.ukf.cov = np.diag([0.01, 0.01, 0.01, 1.0, 1.0])

            # Logging starts after the filter becomes active
            self.logging_start_time = current_time
            return

        dt = current_time - self.last_time
        if dt <= 0:
            return

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vtheta = msg.twist.twist.angular.z

        self.ukf.predict(dt, dx=vx * dt, dy=vy * dt, dtheta=vtheta * dt)

        self.noisy_pose = (x, y)

        self.last_time = current_time
        self.last_state = np.array([x, y, theta])

        self.publish_estimated(msg.header.stamp)


    # ----------------------------------------------------------------------
    # Landmark observations → UKF Update
    # ----------------------------------------------------------------------
    def landmark_callback(self, msg: PointCloud2):
        if self.last_state is None:
            return

        robot_x, robot_y, robot_theta = self.last_state
        cos_t, sin_t = math.cos(robot_theta), math.sin(robot_theta)

        for p in read_points(msg, field_names=("x", "y", "id"), skip_nans=True):
            lx_w, ly_w, lm_id = p

            dx = lx_w - robot_x
            dy = ly_w - robot_y

            obs_x_r = cos_t * dx + sin_t * dy
            obs_y_r = -sin_t * dx + cos_t * dy

            try:
                self.ukf.update((obs_x_r, obs_y_r, int(lm_id)))
            except Exception as e:
                self.get_logger().warn(f"UKF update failed: {e}")

        # Single CSV logging event per callback
        if self.logging_active() and self.gt_pose and self.noisy_pose:
            est = self.ukf.state
            now = self.get_clock().now().nanoseconds * 1e-9

            self.csv_writer.writerow([
                now,
                self.gt_pose[0], self.gt_pose[1],
                est[0], est[1],
                self.noisy_pose[0], self.noisy_pose[1]
            ])
            self.csv_file.flush()

        self.publish_estimated(msg.header.stamp)


    # ----------------------------------------------------------------------
    # Quaternion → yaw conversion
    # ----------------------------------------------------------------------
    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)


    # ----------------------------------------------------------------------
    # Publish UKF estimated odometry
    # ----------------------------------------------------------------------
    def publish_estimated(self, timestamp):
        est = self.ukf.state

        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = "map"

        odom.pose.pose.position.x = est[0]
        odom.pose.pose.position.y = est[1]

        q = Quaternion()
        q.w = math.cos(est[2] / 2)
        q.z = math.sin(est[2] / 2)
        odom.pose.pose.orientation = q

        odom.twist.twist.linear.x = est[3]
        odom.twist.twist.linear.y = est[4]

        self.estimated_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = PositioningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
