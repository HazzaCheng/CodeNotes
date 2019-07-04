#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-07-04

import matplotlib.pyplot as plt
import numpy as np
from dnn_app_utils import *

if __name__ == '__main__':
    # set default configurations of plots
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(1)

