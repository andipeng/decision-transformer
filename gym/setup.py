import os
import sys
from setuptools import setup, find_packages

print("Installing decision transformer (modified)")

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='decision_transformer',
    description='decision transformer gym',
    long_description=read('readme-gym.md'),
    author='Andi Peng',
    install_requires=[
        'click', 'gym>=0.13', 'mujoco-py<2.1,>=2.0', 'termcolor', 'mjrl', 'mj_envs',
    ],
)
