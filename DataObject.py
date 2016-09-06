# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import pandas

from sklearn.preprocessing import LabelEncoder,StandardScaler,\
    MultiLabelBinarizer

from sklearn.preprocessing import Imputer
IdCols = ["eid","tid"]
CategoryCols =["dept","forcredit","attendance","textbookuse","interest",
               "grade","online","profgender"]
ArrCols = ["tags"]
DateCols = ["date"]
NumericalCols = ["helpcount","nothelpcount","profhotness"]

def GetComments(frame):
    raw = frame['comments']
    return raw.replace(np.nan,' ', regex=True)

def _GenericConverterFit(Func,Column,**kwargs):
    Converter = Func(**kwargs)
    Converter.fit(Column.reshape(-1,1))
    return Converter

def GetCategoryColumnConverter(Column):
    return _GenericConverterFit(LabelEncoder,Column)

def GetNumericalColumnConverter(Column):
    return _GenericConverterFit(StandardScaler,Column)

def GetArrayColumnConverter(Column):
    return _GenericConverterFit(MultiLabelBinarizer,Column)

def FitDataPreprocessors(Frame):
    ToRet = []
    Cols = [CategoryCols,NumericalCols,ArrCols]
    Funcs = [GetCategoryColumnConverter,GetNumericalColumnConverter,
             GetArrayColumnConverter]
    for ListOfCols,Func in zip(Cols,Funcs):
        ToRet.extend([Func(Frame[c]) for c in ListOfCols])
    return ToRet

def PreProcessData(PreProcessors,Frame):
    AllCols = CategoryCols + NumericalCols + ArrCols
    Final = []
    NumCols = 0
    for c,p in zip(AllCols,PreProcessors):
        Columns = p.transform(Frame[c])
        Final.append(Columns)
        NumCols += 1 if len(Columns.shape)==1 else Columns.shape[1]
    return Final.reshape(-1,NumCols)
    
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
