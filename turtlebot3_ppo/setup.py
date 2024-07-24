from setuptools import find_packages, setup

package_name = 'turtlebot3_ppo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jblevins32',
    maintainer_email='jacob.blevins@gatech.edu',
    description='Package for training PPO on TurtleBot3 in ROS2 Humble',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = turtlebot3_ppo.train:main',
            'turtlebot3_env = turtlebot3_ppo.turtlebot3_env:main',
            'deploy = turtlebot3_ppo.deploy:main',
        ],
    },
)
