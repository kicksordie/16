# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
from pandas import read_csv

"""
Raw classes for reading in a review
"""

def ReadReviews(Filename):
    """
    Given a csv fille formatted as we want, returns an array of reviews

    Args:
        Filename: path to file
    Returns:
        panda data frame
    """
    frame = read_csv(Filename,
                     header=0)
    return frame


