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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Imputer

from sklearn.linear_model import LinearRegression

def GetData(FileName):
    """
    Returns the pandas dataframe associated with FileName
    """
    RawReview = ReadReviews(FileName)
    return RawReview


def ToVectorizableData(DataFrame):
    """
    Given a data frame, gets the columns we want to use for vectorizing
    
    Args:
        DataFrame: Return from GetData
    Returns:
        Vectorizable data frame
    """
    # XXX for now, ignore the comments
    columns = set(DataFrame.columns) - set(["tags","quality",
                                            "easiness","clarity","helfullness"])
    ToVectorize = DataFrame[list(columns)].to_dict('records')
    return ToVectorize

def FitVectorizer(DataFrame):
    """
    Fits a vectorizer to a dataframe
    
    Args:
         Dataframe: pandas dataa frame to use
    Returns:
         vectorizer
    """
    ToVectorize = ToVectorizableData(DataFrame)
    Vect = DictVectorizer()
    Vect.fit(ToVectorize)
    return Vect

def SafeInput(X):
    """
    Given a feature matri, makes it fittable (ie: no nans)
    
    Args:
         X: feature matrix
    Returns:
         fitted and transformed (nan-free) feature matrix
    """
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return imp.fit_transform(X)

def FitTraining(Features,Labels):
    """
    Fit the training data to the model we want
    
    Args:
        Features: matrix of NxF for fitting
        Labels: to fit for each matrix
    Returns:
        fitted model object
    """
    lr = LinearRegression()
    lr.fit(SafeInput(Features),Labels)
    return lr

def GetLabel(DataFrame):
    """
    Gets the label (y) associated with the data frame
    """
    return DataFrame['quality']

def WritePredictions(Frame,Predictions):
    """
    Given a data frame (with ids) and predictions, write out the predictions 
    file 

    Args:
         Frame: data frame for the predictions, assumed conserved order
         Predictions: predicted values for each 'row' in frame
    """
    with open('predictions.csv', 'w') as f:
        f.write("id,quality\n")
        for row_id, prediction in zip(Frame['id'], Predictions):
            f.write('{},{}\n'.format(row_id, prediction))

def FitCommentsVectorizer(Frame):
    return TfidfVectorizer.fit(Frame,maxdf=0.7,mindf=0.1)

def ConvertCommentsToTfIdf(Frame,Vectorizer):
    Copy = Frame.copy()
    Copy["comments"] = Vectorizer.transform(Copy["comments"])
    return Copy


def run():
    """
    Reads in the data, splits it into training and testing, transforms it 
    """
    TrainFile = "../data/train.csv"
    TestFile = "../data/test.csv"
    ForceRead=False
    ForceVect=False
    # Read in the data sets
    FractionTrain = 0.8
    # split into validation and training
# stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    AllTraining = CheckpointUtilities.getCheckpoint("Train.pkl",GetData,
                                                    ForceRead,TrainFile)
    NTrain = AllTraining.shape[0]
    TrainMask = np.random.rand(NTrain) < FractionTrain
    TrainData = AllTraining[TrainMask]
    ValidData = AllTraining[~TrainMask]
    # fit to the comments on the training data
    #TfIdf = FitCommentsVectorizer(TrainData)
    TestData = CheckpointUtilities.getCheckpoint("Test.pkl",GetData,
                                                 ForceRead,TestFile)
    # fit a vecotrizer to the training set
    Vect = CheckpointUtilities.getCheckpoint("Vect.pkl",FitVectorizer,
                                             ForceVect,TrainData)
    # transform the data sets to feature matrices
    TrainFeatures = Vect.transform(ToVectorizableData(TrainData))
    ValidFeatures = Vect.transform(ToVectorizableData(ValidData))
    TestFeatures = Vect.transform(ToVectorizableData(TestData))
    # train our model, using the training data
    TrainLabels =GetLabel(TrainData)
    ValidLabels = GetLabel(ValidData)
    lr = FitTraining(TrainFeatures,TrainLabels)
    Sanitize = lambda x : np.maximum(2,np.minimum(10,x))
    PredValid = Sanitize(lr.predict(SafeInput(ValidFeatures)))
    PredTrain= Sanitize(lr.predict(SafeInput(TrainFeatures)))
    PredTest= Sanitize(lr.predict(SafeInput(TestFeatures)))
    WritePredictions(TestData,PredTest)
    # make a very simple diagnostic figure
    fig = PlotUtilities.figure()
    Limits = [1,11]
    plt.plot(TrainLabels,PredTrain,'r.')
    plt.plot(ValidLabels,PredValid,'b.')
    plt.xlim(Limits)
    plt.ylim(Limits)
    PlotUtilities.lazyLabel("Actual","Predicted","RMP Scores")
    PlotUtilities.savefig(fig,"out.png")
    

if __name__ == "__main__":
    run()
