import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/felix/Schreibtisch/projects/ros2_homework_1/install/kalman_positioning_py'
