"""
Deep learning, GAN based method to generate target frames (combined to get vids at the end) from the target pose skeletons.
Should perform after step A & B.
"""
import os
import numpy as np
import torch
import time
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable
from pathlib import Path
import sys


pix2pixhd_dir = Path('../src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))