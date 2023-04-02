import math
import numpy as np

from _func import *

root = str(Path(__file__).parent) + "\\"

time_limit = 40
T, C = [6, 6, 8, 9], [1, 1, 2, 3]

# apt: Aperiodic Task
apt_time, apt_jobs, apt_dls = [3, 22], [2, 3], [8, 5]
apt_number = 2  # The 3rd task

# Start Coding!