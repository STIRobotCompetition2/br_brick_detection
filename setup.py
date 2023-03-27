from setuptools import setup
import os
from glob import glob


package_name = 'br_brick_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, "launch"), glob('launch/**')),
        (os.path.join('share', package_name, "config"), glob('config/**')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='svenbecker',
    maintainer_email='sven.becker@epfl.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_brick_detection = br_brick_detection.yolo_brick_detector:main'
        ],
    },
)
