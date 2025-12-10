import math
import numpy as np

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
    UKF-Localization Node für 2D Mobile Robot:
        Prediction:   Odometry
        Update:       Landmark observations
    """

    def __init__(self):
        super().__init__("kalman_positioning_node")

        self.get_logger().info("Initializing Kalman Positioning Node…")

        # ===========================================================
        # LOAD LANDMARKS
        # ===========================================================
        self.landmark_manager = LandmarkManager()
        ok = self.landmark_manager.load_from_csv("src/kalman_positioning/landmarks.csv")

        if not ok:
            self.get_logger().error("Could not load landmarks! UKF will not work")

        # ===========================================================
        # INIT UKF
        # ===========================================================
        self.ukf = UKF(
            process_noise_xy=0.05,
            process_noise_theta=0.01,
            measurement_noise_xy=0.05,
            num_landmarks=len(self.landmark_manager.get_all_landmarks())
        )

        # Fülle Landmark-Tabelle für UKF
        for i, (lm_id, (lx, ly)) in enumerate(self.landmark_manager.get_all_landmarks().items()):
            self.ukf.landmarks[i] = [lm_id, lx, ly]

        self.is_first_odom = True
        self.last_time = None
        self.last_state = None  # [x,y,theta]

        # ===========================================================
        # Subscribers
        # ===========================================================
        self.create_subscription(
            Odometry,
            "/robot_noisy",
            self.odometry_callback,
            10
        )

        self.create_subscription(
            PointCloud2,
            "/landmarks_observed",
            self.landmarks_observed_callback,
            10
        )

        # ===========================================================
        # Publisher
        # ===========================================================
        self.estimated_odom_pub = self.create_publisher(
            Odometry,
            "/robot_estimated_odometry",
            10
        )

        self.get_logger().info("Kalman Positioning Node initialized successfully.")

    # ======================================================================
    # ODOMETRY (PREDICTION)
    # ======================================================================
    def odometry_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)

        current_time = (msg.header.stamp.sec +
                        msg.header.stamp.nanosec * 1e-9)

        if self.is_first_odom:
            self.last_time = current_time
            self.last_state = np.array([x, y, theta])
            self.is_first_odom = False
            return

        dt = current_time - self.last_time
        if dt <= 0:
            return

        # Odom-basierte Bewegungsdifferenzen
        dx = x - self.last_state[0]
        dy = y - self.last_state[1]
        dtheta = self.normalize_angle(theta - self.last_state[2])

        # UKF PREDICTION
        self.ukf.predict(dt=dt, dx=dx, dy=dy, dtheta=dtheta)

        # Update stored values
        self.last_time = current_time
        self.last_state = np.array([x, y, theta])

        # Publish estimation
        self.publish_estimated_odometry(msg.header.stamp)

    # ======================================================================
    # LANDMARKS (UPDATE)
    # ======================================================================
    def landmarks_observed_callback(self, msg: PointCloud2):

        for p in read_points(msg, field_names=("x", "y", "id"), skip_nans=True):
            obs_x, obs_y, lm_id = p

            # eine einzelne Landmark-Observation → UKF.update()
            try:
                self.ukf.update((obs_x, obs_y, int(lm_id)))
            except Exception as e:
                self.get_logger().warn(f"Update failed for lm {lm_id}: {e}")

        # nach dem Update sofort publizieren
        self.publish_estimated_odometry(msg.header.stamp)

    # ======================================================================
    # HELPER
    # ======================================================================
    def quaternion_to_yaw(self, q: Quaternion):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return self.normalize_angle(math.atan2(siny_cosp, cosy_cosp))

    def normalize_angle(self, angle: float):
        return (angle + math.pi) % (2*math.pi) - math.pi

    # ======================================================================
    # PUBLISHING
    # ======================================================================
    def publish_estimated_odometry(self, timestamp):
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "robot_estimated"

        x = self.ukf.x_

        # Position
        odom.pose.pose.position.x = x[0]
        odom.pose.pose.position.y = x[1]

        # Orientation aus theta erzeugen
        q = self.yaw_to_quaternion(x[2])
        odom.pose.pose.orientation = q

        # Velocity
        odom.twist.twist.linear.x = x[3]
        odom.twist.twist.linear.y = x[4]

        # Covariance 5×5 → 6×6
        cov = np.zeros((6, 6))
        P = self.ukf.P_

        cov[0, 0] = P[0, 0]  # x
        cov[1, 1] = P[1, 1]  # y
        cov[5, 5] = P[2, 2]  # theta

        cov[0, 1] = cov[1, 0] = P[0, 1]
        cov[0, 5] = cov[5, 0] = P[0, 2]
        cov[1, 5] = cov[5, 1] = P[1, 2]

        odom.pose.covariance = cov.flatten().tolist()

        self.estimated_odom_pub.publish(odom)

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.w = math.cos(yaw / 2)
        q.z = math.sin(yaw / 2)
        q.x = 0.0
        q.y = 0.0
        return q


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
