#!/bin/bash
#                       Scirpt documentation
#   This script is to create a python venv for standalone running.
#   The extra package is only for running benchmark
#
#   The target packages are:
#       numpy
#       matplotlib   -  for visualize the route result in testing routine
#

# remove the venv package
rm -rf ~/venv

# create the venv environment
virtualenv -p /usr/bin/python3 ~/venv

# activate the environment
source ~/venv/bin/activate

# Update pip
pip install --upgrade pip

# install packages in binary format (wheel)
pip install numpy
pip install matplotlib

