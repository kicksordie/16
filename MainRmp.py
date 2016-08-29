# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from GeneralUtil.python import CheckpointUtilities,PlotUtilities
from src.DataIo import ReadReviews
from src.DataObject import RatingsObject
# sklearn stuff
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer

from sklearn.linear_model import LinearRegression

def GetData(FileName):
    RawReview = ReadReviews(FileName)
    return RawReview


def ToVectorizableData(DataFrame):
    # XXX for now, ignore the comments
    columns = set(DataFrame.columns) - set(["tags","quality","comments",
                                            "easiness","clarity","helfullness"])
    ToVectorize = DataFrame[list(columns)].to_dict('records')
    return ToVectorize

def FitVectorizer(DataFrame):
    ToVectorize = ToVectorizableData(DataFrame)
    Vect = DictVectorizer()
    Vect.fit(ToVectorize)
    return Vect

def SafeInput(X):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return imp.fit_transform(X)

def FitTraining(Features,Labels):
    lr = LinearRegression()
    lr.fit(SafeInput(Features),Labels)
    return lr

    

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    TrainFile = "../data/train.csv"
    TestFile = "../data/test.csv"
    ForceRead=False
    ForceVect=False
    TrainData = CheckpointUtilities.getCheckpoint("Train.pkl",GetData,
                                                  ForceRead,TrainFile)
    TestData = CheckpointUtilities.getCheckpoint("Test.pkl",GetData,
                                                 ForceRead,TestFile)

    Vect = CheckpointUtilities.getCheckpoint("Vect.pkl",FitVectorizer,
                                             ForceVect,TrainData)
    TrainFeatures = Vect.transform(ToVectorizableData(TrainData))
    TestFeatures = Vect.transform(ToVectorizableData(TestData))
    TrainLabels = TrainData['quality']
    lr = FitTraining(TrainFeatures,TrainLabels)
    pred = lr.predict(SafeInput(TestFeatures))

if __name__ == "__main__":
    run()
