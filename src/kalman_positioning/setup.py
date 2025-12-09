from setuptools import find_packages, setup

package_name = 'kalman_positioning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/positioning.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Felix Diehl',
    maintainer_email='felix.diehlsw@gmail.com',
    description='Python implementation of Kalman positioning for ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'positioning_node = kalman_positioning.positioning_node:main',
        ],
    },
)
