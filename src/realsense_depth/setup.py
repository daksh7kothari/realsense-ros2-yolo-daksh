from glob import glob

from setuptools import setup

package_name = 'realsense_depth'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daksh',
    maintainer_email='dakshkothari77@gmail.com',
    description='Depth probing and point cloud analysis for an Intel RealSense D435i.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_measure = realsense_depth.depth_measure:main',
            'cloud_pipeline = realsense_depth.cloud_pipeline:main',
            'yolo_measure = realsense_depth.yolo_measure:main',
        ],
    },
)
