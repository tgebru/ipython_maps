
# coding: utf-8

# In[ ]:

def prepare_features(lines,features):
    labels=[]
    for l in lines:
      fname=l.split(' ')[0]
      labels.append(int(l.split(' ')[1].strip()))

    feat_list=[f.tolist() for f in features[0:len(labels)]]
    return labels,feat_list


# In[ ]:

def scale_feats(features, feat_min,feat_max):
    #Scale features to be between 0-1
    feat_mat=np.array(features)
    delta=feat_max-feat_min
    feat_scaled=(feat_mat-feat_min)/feat_max 
    feat_scaled[np.isnan(feat_scaled)]=0
   
    return feat_scaled


# In[148]:

#Train an SVM using features from CNN
import sys
import os
import pickle
sys.path.insert(1, '/afs/cs.stanford.edu/u/tgebru/software/liblinear-2.1/python')
from liblinearutil import *
import numpy as np
import copy

FEAT='fc7'

data_root='/imagenetdb3/tgebru/cvpr2016/housing_data/ny_housing'
train_files=os.path.join(data_root,'housing_2013_class_train.txt')
train_lines=open(train_files,'rb').readlines()

val_files=os.path.join(data_root,'housing_2013_class_val.txt')
val_lines=open(val_files,'rb').readlines()

train_feat_pname='/imagenetdb3/tgebru/cvpr2016/housing_train_2013_class_%s'%FEAT
val_feat_pname='/imagenetdb3/tgebru/cvpr2016/housing_val_2013_class_%s'%FEAT

print 'loading train features...'
with open(train_feat_pname,'rb') as f:
  train_feats = pickle.load(f)

print 'loading val features...'
with open(val_feat_pname,'rb') as f:
  val_feats = pickle.load(f)

#Scale features
print 'Scaling features...'
train_feat_mat=np.array(train_feats)
train_feat_max=train_feat_mat.max(0)
train_feat_min=train_feat_mat.min(0)
train_feats_scaled=scale_feats(train_feats,train_feat_min,train_feat_max) 
val_feats_scaled=scale_feats(val_feats,train_feat_min,train_feat_max)

#Prepare labels and features
print 'Preparing features and labels for training...'
train_labels,train_feat_list=prepare_features(train_lines,train_feats_scaled)
val_labels,val_feat_list=prepare_features(val_lines,val_feats_scaled)


# In[149]:

print 'training model with grid search for parameters...'

[best_c, acc] = train(train_labels + val_labels, 
          train_feat_list + val_feat_list, '-C -B 1')

m=train(train_labels,train_feat_list, '-C %d'%best_c)
print 'evaluating model...'
#p_label, p_acc, p_val = predict(labels,feat_list, m)
#print p_label, p_acc, p_val
save_model('/imagenetdb3/tgebru/cvpr2016/housing_val_2013_class_%s.model'%FEAT,m)
