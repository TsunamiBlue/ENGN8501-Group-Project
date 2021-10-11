"""
Deep learning, testing the entire steps.
Should feed the same person as the target person.
"""
import os
import numpy as np
import torch
import time
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable