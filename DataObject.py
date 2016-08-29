# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import pandas

def CategoryCastToInteger(column):
    casted = pandas.Series(column,dtype="category")
    print(casted.cat)
    return casted

CategoryCols = ['dept','attendance','textbookuse','interest','grade','online',
                'profgender','profhotness','tags']


class RatingsObject:
    """
    This is a more abstracted review object; we take in a 'raw' review as
    strings, and then convert the values to appropriate types
    """
    def __init__(self,Review):
        self.Copy = copy.deepcopy(Review)
        Columns = self.Copy.columns
        # copy the original columns
        for c in Columns:
            setattr(self,c,self.Copy[c])
