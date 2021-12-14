import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
#import RandomBinningFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt 
#%matplotlib inline 
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean
#import seaborn as sns

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rcParams['mathtext.fontset'] = 'cm'

## for LaTeX typefont
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

## for another LaTeX typefont
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# rc('text', usetex = True)

print("Packages Loaded!!!")


# # Read Frequency Vector

# In[ ]:

data_path = "/olga-data0/Sarwan/GISAID_Data/"
  


variant_names_1 = np.load(data_path + "0_1000000_variants_names.npy")

variant_orig = []

for e in range(len(variant_names_1)):
    variant_orig.append(variant_names_1[e])


variant_names_1 = np.load(data_path + "1000000_2000000_variants_names.npy")

for e in range(len(variant_names_1)):
    variant_orig.append(variant_names_1[e])

variant_names_1 = np.load(data_path + "2000000_3000000_variants_names.npy")

for e in range(len(variant_names_1)):
    variant_orig.append(variant_names_1[e])
    
variant_names_1 = np.load(data_path + "3000000_4072342_variants_names.npy")

for e in range(len(variant_names_1)):
    variant_orig.append(variant_names_1[e])
    
    
print("Attributed data Reading Done, length ==>>",len(variant_orig), ", Expected ==>>", str(4072342))


# In[14]:


unique_varaints = list(np.unique(variant_orig))





name_variant_reduced = []

for ind_reduced in range(len(variant_orig)):
    if variant_orig[ind_reduced]=="B.1.1.7" or variant_orig[ind_reduced]=="B.1.617.2" or variant_orig[ind_reduced]=="AY.4" or variant_orig[ind_reduced]=="B.1.2" or variant_orig[ind_reduced]=="B.1" or variant_orig[ind_reduced]=="B.1.177"  or variant_orig[ind_reduced]=="P.1" or variant_orig[ind_reduced]=="B.1.1" or variant_orig[ind_reduced]=="B.1.429"  or variant_orig[ind_reduced]=="AY.12" or variant_orig[ind_reduced]=="B.1.160" or variant_orig[ind_reduced]=="B.1.526" or variant_orig[ind_reduced]=="B.1.1.519" or variant_orig[ind_reduced]=="B.1.351" or variant_orig[ind_reduced]=="B.1.1.214"  or variant_orig[ind_reduced]=="B.1.427" or variant_orig[ind_reduced]=="B.1.221" or variant_orig[ind_reduced]=="B.1.258" or variant_orig[ind_reduced]=="B.1.177.21" or variant_orig[ind_reduced]=="D.2" or variant_orig[ind_reduced]=="B.1.243"  or variant_orig[ind_reduced]=="R.1":
        name_variant_reduced.append(variant_orig[ind_reduced])


y_orig = np.array(name_variant_reduced)

np.save(data_path + "top_attributes_only_information.npy",y_orig)

print("All Processing Done!!!")
