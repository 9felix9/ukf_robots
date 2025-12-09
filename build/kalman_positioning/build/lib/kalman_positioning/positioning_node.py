import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

class PositioningNode(Node):
    def __init__(self):
        super().__init__('kalman_positioning_node')
        self.get_logger().info('Initializing Kalman Positioning Node')

        # Subscribers
        self.create_subscription(Odometry, '/robot_noisy', self.odometry_callback, 10)
        self.create_subscription(PointCloud2, '/landmarks_observed', self.landmarks_callback, 10)

        # Publisher
        self.odom_publisher = self.create_publisher(Odometry, '/robot_estimated_odometry', 10)

        # Kalman filter state (placeholder)
        self.state = None  # Placeholder for Kalman filter state
        self.covariance = None  # Placeholder for covariance matrix

    def odometry_callback(self, msg):
        """
        Callback for noisy odometry measurements

        STUDENT TODO:
        1. Extract position (x, y) and orientation (theta) from the message
        2. Update the Kalman filter's prediction step with this odometry
        3. Publish the estimated odometry
        """
        self.get_logger().debug(f'Received odometry: x={msg.pose.pose.position.x}, y={msg.pose.pose.position.y}')

    def landmarks_callback(self, msg):
        """
        Callback for landmark observations

        STUDENT TODO:
        1. Parse the PointCloud2 data to extract landmark observations
        2. Update the Kalman filter's measurement update step with these observations
        3. Optionally publish the updated estimated odometry
        """
        self.get_logger().debug('Received landmark observations')

def main(args=None):
    rclpy.init(args=args)
    node = PositioningNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()