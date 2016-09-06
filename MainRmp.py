# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from GeneralUtil.python import CheckpointUtilities,PlotUtilities
from src.DataIo import ReadReviews
from src.DataObject import RatingsObject,GetComments
# sklearn stuff
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Imputer

from sklearn.linear_model import LinearRegression

# Output the Mean Squared Error using our held out training data
from sklearn.metrics import mean_squared_error

def GetData(FileName):
    """
    Returns the pandas dataframe associated with FileName
    """
    RawReview = ReadReviews(FileName)
    return RawReview


def FeatureFitter(DataFrame,max_features=1000,min_df=0.01,max_df=0.95):
    """
    Fits a vectorizer to a dataframe
    
    Args:
         Dataframe: pandas dataa frame to use
    Returns:
         list of pre-processors in the order DataObject understands
    """
    Comments = GetComments(DataFrame)
    vect = TfidfVectorizer(max_features=max_features,min_df=min_df,
                           max_df=max_df)
    vect.fit(Comments)
    return vect
    
def TransformToFeatures(Fitter,DataFrame):
    vocab = Fitter
    return vocab.transform(GetComments(DataFrame))

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
    lr.fit(Features,Labels)
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
        for row_id, prediction in zip(Frame['eid'], Predictions):
            f.write('{},{}\n'.format(row_id, prediction))

def GetScore(Predicted,Actual):
    return mean_squared_error(Actual,Predicted)

def run():
    """
    Reads in the data, splits it into training and testing, transforms it 
    """
    BaseData = "../data/"
    TrainFile = BaseData + "train.csv"
    TestFile = BaseData + "test.csv"
    ForceRead=False
    ForceVect=False
    # Read in the data sets
    FractionTrain = 0.9
    np.random.seed(42)
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
    PreProcessors = CheckpointUtilities.getCheckpoint("Vect.pkl",FeatureFitter,
                                                      ForceVect,TrainData)
    Data = [TrainData,ValidData,TestData]
    TrainFeatures,ValidFeatures,TestFeatures = \
        [TransformToFeatures(PreProcessors,d) for d in Data]
    # train our model, using the training data
    TrainLabels =GetLabel(TrainData)
    ValidLabels = GetLabel(ValidData)
    lr = FitTraining(TrainFeatures,TrainLabels)
    Sanitize = lambda x : np.maximum(2,np.minimum(10,x))
    PredValid = Sanitize(lr.predict(ValidFeatures))
    PredTrain= Sanitize(lr.predict(TrainFeatures))
    PredTest= Sanitize(lr.predict(TestFeatures))
    WritePredictions(TestData,PredTest)
    # make a very simple diagnostic figure
    fig = PlotUtilities.figure()
    Limits = [1,11]
    ValidRMSE = GetScore(PredValid,ValidLabels)
    TrainRMSE = GetScore(PredTrain,TrainLabels)
    ValidLabel = "Valid RMSE: {:.2f}".format(ValidRMSE)
    TrainLabel = "Train RMSE: {:.2f}".format(TrainRMSE)
    plt.plot(TrainLabels,PredTrain,'r.',label=TrainLabel)
    plt.plot(ValidLabels,PredValid,'b.',label=ValidLabel)
    plt.xlim(Limits)
    plt.ylim(Limits)
    PlotUtilities.lazyLabel("Actual","Predicted","RMP Scores",frameon=True)
    PlotUtilities.savefig(fig,"out.png")
    

if __name__ == "__main__":
    run()
