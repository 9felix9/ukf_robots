import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from geometry_msgs.msg import Quaternion
from ukf import UKF
import numpy as np
import math


class PositioningNode(Node):
    """
    @brief
    Positioning node for UKF-based robot localization (Student Assignment).

    This node subscribes to:a
      - /robot_noisy: Noisy odometry (dead-reckoning)
      - /landmarks_observed: Noisy landmark observations

    And publishes to:
      - /robot_estimated_odometry: Estimated pose and velocity from the filter

    @details
    STUDENT ASSIGNMENT:
    Implement the Kalman filter logic (e.g., UKF) to fuse odometry and
    landmark observations to estimate the robot's true position.

    Expected UKF state vector:
        x = [x, y, theta, vx, vy]

    Students must implement:
    - State initialization
    - Prediction step using noisy odometry
    - Measurement update using landmark observations
    - State covariance propagation
    """

    def __init__(self):
        super().__init__("kalman_positioning_node")
        self.get_logger().info("Initializing Kalman Positioning Node")

        # ----------------------------------------------------------------------
        # SUBSCRIBERS
        # ----------------------------------------------------------------------
        self.create_subscription(
            Odometry,
            "/robot_noisy",
            self.odometry_callback,
            10
        )
        self.get_logger().info("Subscribed to /robot_noisy")

        self.create_subscription(
            PointCloud2,
            "/landmarks_observed",
            self.landmarks_observed_callback,
            10
        )
        self.get_logger().info("Subscribed to /landmarks_observed")

        # ----------------------------------------------------------------------
        # PUBLISHER
        # ----------------------------------------------------------------------
        self.estimated_odom_pub = self.create_publisher(
            Odometry,
            "/robot_estimated_odometry",
            10
        )
        self.get_logger().info("Publishing to /robot_estimated_odometry")

        self.get_logger().info("Kalman Positioning Node initialized successfully")

        # ----------------------------------------------------------------------
        # PLACEHOLDER: KALMAN FILTER INTERNAL STATE
        # ----------------------------------------------------------------------
        """
        STUDENT TODO:
        Initialize UKF state variables here:
            self.x = np.zeros(5)
            self.P = np.eye(5)
            self.Q = ...
            self.R = ...

        Students should also create the sigma point generator and
        update logic used by the UKF.
        """

        # hier utf initialisieren und irgendwie die Werte übergeben, die in der positioning.launch.py 
        # als default werte gesetzt werden. Zudem auch den landmark manager nutzen um die landmark liste
        # zu bekommen. Diese liste wird dann an ukf.landmarks_ oder so übergeben. 
        # Diese Liste kann dann von der methode measurement model genutzt werden
        self.ukf = UKF(
            
        )


    # ==========================================================================
    # CALLBACK FUNCTIONS
    # ==========================================================================

    def odometry_callback(self, msg: Odometry):
        """
        @brief
        Callback for noisy odometry measurements (Prediction Step input).

        @details
        STUDENT TODO:
        1. Extract position (x, y) and orientation (theta) from the message.
        2. Extract linear velocities (vx, vy).
        3. Use these values in the UKF PREDICTION step.
        4. Replace the current placeholder and publish estimated odometry.
        """

        self.get_logger().debug(
            f"Odometry received: x={msg.pose.pose.position.x:.3f}, "
            f"y={msg.pose.pose.position.y:.3f}"
        )

        # Extract pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.quaternion_to_yaw(msg.pose.pose.orientation)

        # Extract twist
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        self.get_logger().debug(
            f"Parsed: x={x:.3f}, y={y:.3f}, theta={theta:.3f}, "
            f"vx={vx:.3f}, vy={vy:.3f}"
        )

        # ------------------------------------------------------------------
        # STUDENT TODO:
        # Use (x, y, theta, vx, vy) to update UKF PREDICTION step.
        # ------------------------------------------------------------------

        # TEMPORARY PLACEHOLDER (publishes noisy odometry):
        self.publish_estimated_odometry(msg.header.stamp, msg)

    def landmarks_observed_callback(self, msg: PointCloud2):
        """
        @brief
        Callback for noisy landmark observations (Measurement Step input).

        @details
        STUDENT TODO:
        1. Parse the PointCloud2 message.
        2. For each landmark:
            - Extract ID, observed position (x, y)
            - Perform the UKF UPDATE step using the measurement model.
        """

        self.get_logger().debug(
            f"Landmark observation received with {msg.width} points"
        )

        # Attempt to parse the landmark observations
        try:
            count = 0
            for p in read_points(msg, field_names=("x", "y", "id"), skip_nans=True):
                obs_x, obs_y, landmark_id = p

                self.get_logger().debug(
                    f"Landmark {int(landmark_id)} observed at "
                    f"({obs_x:.3f}, {obs_y:.3f})"
                )
                count += 1

                # ----------------------------------------------------------
                # STUDENT TODO:
                # Run UKF measurement update using:
                #   - landmark_id
                #   - observed position (obs_x, obs_y)
                # ----------------------------------------------------------

            self.get_logger().debug(
                f"Processed {count} landmark observations"
            )

        except Exception as e:
            self.get_logger().warn(
                f"Failed to parse landmark observations: {e}"
            )

    # ==========================================================================
    # HELPER FUNCTIONS
    # ==========================================================================

    def quaternion_to_yaw(self, q: Quaternion):
        """
        @brief Convert quaternion orientation to yaw angle [-pi, pi].

        Equivalent to tf2::Matrix3x3(q).getRPY(roll, pitch, yaw) in C++.
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return self.normalize_angle(yaw)

    def normalize_angle(self, angle: float):
        """
        @brief Normalize any angle to the range [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def publish_estimated_odometry(self, timestamp, odom_msg: Odometry):
        """
        @brief
        Publish the estimated odometry after filtering.

        @details
        STUDENT TODO:
        Replace placeholder with real estimated pose and covariance
        from your UKF.

        For now, we forward the noisy odometry as a placeholder.
        """

        estimated = Odometry()
        estimated.header.stamp = timestamp
        estimated.header.frame_id = "map"
        estimated.child_frame_id = "robot_estimated"

        # Placeholder, copy noisy values:
        estimated.pose = odom_msg.pose
        estimated.twist = odom_msg.twist

        self.estimated_odom_pub.publish(estimated)

# ==========================================================================
# MAIN
# ==========================================================================

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
