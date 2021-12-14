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


data_path = "/olga-data0/Sarwan/GISAID_Data/"
  
k_mers_1 = np.load(data_path + "0_1000000_minimizers_k_9_m_3_then_k_mers_3_file.npy")

frequency_vector_read_final = []

for e in range(len(k_mers_1)):
    frequency_vector_read_final.append(k_mers_1[e])
    
k_mers_1 = np.load(data_path + "1000000_2000000_minimizers_k_9_m_3_then_k_mers_3_file.npy")

for e in range(len(k_mers_1)):
    frequency_vector_read_final.append(k_mers_1[e])
    
k_mers_1 = np.load(data_path + "2000000_3000000_minimizers_k_9_m_3_then_k_mers_3_file.npy")

for e in range(len(k_mers_1)):
    frequency_vector_read_final.append(k_mers_1[e])
    
k_mers_1 = np.load(data_path + "3000000_4072342_minimizers_k_9_m_3_then_k_mers_3_file.npy")

for e in range(len(k_mers_1)):
    frequency_vector_read_final.append(k_mers_1[e])

print("Frequency Vector length ==>>",len(frequency_vector_read_final), ", Expected ==>>", str(4072342))

print("Single Feature Vector Length ==>>",len(frequency_vector_read_final[0]))




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


# In[18]:


int_variants = []
for ind_unique in range(len(variant_orig)):
    variant_tmp = variant_orig[ind_unique]
    ind_tmp = unique_varaints.index(variant_tmp)
    int_variants.append(ind_tmp)
    
print("Attribute data preprocessing Done")


freq_vec_reduced = []
int_variant_reduced = []
name_variant_reduced = []

for ind_reduced in range(len(frequency_vector_read_final)):
    if variant_orig[ind_reduced]=="B.1.1.7" or variant_orig[ind_reduced]=="B.1.617.2" or variant_orig[ind_reduced]=="AY.4" or variant_orig[ind_reduced]=="B.1.2" or variant_orig[ind_reduced]=="B.1" or variant_orig[ind_reduced]=="B.1.177"  or variant_orig[ind_reduced]=="P.1" or variant_orig[ind_reduced]=="B.1.1" or variant_orig[ind_reduced]=="B.1.429"  or variant_orig[ind_reduced]=="AY.12" or variant_orig[ind_reduced]=="B.1.160" or variant_orig[ind_reduced]=="B.1.526" or variant_orig[ind_reduced]=="B.1.1.519" or variant_orig[ind_reduced]=="B.1.351" or variant_orig[ind_reduced]=="B.1.1.214"  or variant_orig[ind_reduced]=="B.1.427" or variant_orig[ind_reduced]=="B.1.221" or variant_orig[ind_reduced]=="B.1.258" or variant_orig[ind_reduced]=="B.1.177.21" or variant_orig[ind_reduced]=="D.2" or variant_orig[ind_reduced]=="B.1.243"  or variant_orig[ind_reduced]=="R.1":
        freq_vec_reduced.append(frequency_vector_read_final[ind_reduced])
        int_variant_reduced.append(int_variants[ind_reduced])
        name_variant_reduced.append(variant_orig[ind_reduced])

print("Total Sequences after reducing data ==>>",len(freq_vec_reduced))
######################################################################
#tem_fre_vec = freq_vec_reduced[0:5000]
#tmp_true_labels = int_variant_reduced[0:5000]
#tmp_true_labels_orig = name_variant_reduced[0:5000]

#X = np.array(tem_fre_vec)
#y =  np.array(tmp_true_labels)
#y_orig =  np.array(tmp_true_labels_orig)
######################################################################

X = np.array(freq_vec_reduced)
y =  np.array(int_variant_reduced)
y_orig = np.array(name_variant_reduced)


from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=1, test_size=0.9)
sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y)) 

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
y_train_orig, y_test_orig = y_orig[train_index], y_orig[test_index]

print("Train-Test Split Done")

    
print("X_train rows = ",len(X_train),"X_train columns = ",len(X_train[0]))
print("X_test rows = ",len(X_test),"X_test columns = ",len(X_test[0]))

#print("Random Fourier Features Starts here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#start_time = time.time()
#rbf_feature = RBFSampler(gamma=1, n_components=500)
#rbf_feature.fit(X_train)
#X_features_train = rbf_feature.transform(X_train)
#X_features_test = rbf_feature.transform(X_test)



print("Data reduction using kernel Done@@@@@@@@@@@@@@")

unique_labels = np.unique(y)
print("unique labels = ",np.unique(y))

# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
aaa = (X_train.shape)
MAX_NB_WORDS = aaa[1]
EMBEDDING_DIM = 500
Memory_Units = 200
epochs_size = 10 # one epoch = one forward pass and one backward pass of all the training examples
batch_size = 100 # the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

#model = Sequential()
#model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=np.array(X).shape[1]))
#model.add(LSTM(Memory_Units, dropout=0.2, recurrent_dropout=0.2))
##model.add(Dense(EMBEDDING_DIM, activation='relu'))
#model.add(Dense(max(unique_labels)+1, activation='softmax'))
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(MAX_NB_WORDS, input_dim=np.array(X).shape[1], activation='relu'))
	model.add(Dense(max(unique_labels)+1, activation='softmax'))
	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

start_time = time.time()
estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs_size, batch_size=batch_size, validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

end_time = time.time() - start_time
print("Keras Classifier Total Time in seconds =>",end_time)

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)
    
RC_acc = metrics.accuracy_score(y_test, y_pred)
RC_prec = metrics.precision_score(y_test, y_pred,average='weighted')
RC_recall = metrics.recall_score(y_test, y_pred,average='weighted')
RC_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
RC_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
RC_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
confuse = confusion_matrix(y_test, y_pred)
y_prob = y_pred
macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_prob, average='macro')

check = [RC_acc,RC_prec,RC_recall,RC_f1_weighted,RC_f1_macro,RC_f1_micro,macro_roc_auc_ovo[1]]

print("Final score",check)
## history = model.fit((X_train), np.array(y_train), epochs=32, batch_size=40,validation_split=0.1)
#history = model.fit(np.array(X_train), np.array(y_train), epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
                                             
#accr = estimator.evaluate(X_test,y_test)
#print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#print(aac,np.mean(recall),np.mean(precision),np.mean(f1),np.mean(f1_macro),np.mean(roc_auc))

print("Keras Classifier Total Time in seconds =>",end_time)

print("All Processing Done!!!")
