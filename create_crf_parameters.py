
# coding: utf-8

# In[98]:

#Create a file with xi,xj, cost_source-->xi, cost_xi-->sink
#CRF connecting points that are geographically close to each other
#Also connecting points 
#File will be read by mincut/maxflow c++ code to construct graph 
import numpy as np
import pickle
import sys
#sys.path.insert(0, '/scr/r6/tgebru/tools/liblinear-2.1/python')
sys.path.insert(0,'/imagenetdb3/tgebru/software/liblinear-2.1/python')
from liblinearutil import *

#FEAT='fc7'
FEAT='prob'
USE_SVM=False
train_feat_pname='/imagenetdb3/tgebru/cvpr2016/housing_train_2013_class_%s'%FEAT
svm_model_name='/imagenetdb3/tgebru/cvpr2016/housing_val_2013_class_%s.model'%FEAT
train_data_pname='/imagenetdb3/tgebru/cvpr2016/housing_train_2013_class_cnn_data.pickle'

#Load features for training labels
print 'loading train features...'
with open(train_feat_pname,'rb') as f:
  train_feats = pickle.load(f)

#Load SVM weights
if USE_SVM:
    print 'loading SVM weights...'
    model=load_model(svm_model_name)
    #Get SVM Weights
    #Decision function=W*x_ithlabel+b_ithlabel
    [w_0, b_0]=model.get_decfun(label_idx=0)
    
#Load lat,lng,actual,predicted labels for training
print 'loading training labels and lat/longs...'
with open(train_data_pname,'rb') as f:
    train_data=pickle.load(f)


# In[ ]:

#We want all the points in a zipcode to have the same value
#Also want nearby zip codes to have similar values?
#How do we incorporate multiple images from the same place?

#Create adjacency matricies for
#Euclidean location distance, feature distance, unary labels

#Create numpy matrix of training data

train_array=np.asarray(train_data,dtype=np.float)

# Euclidean location difference Matrix between all points
loc_mat=np.zeros((train_array.shape[0],train_array.shape[0]))
for i in xrange(train_array.shape[0]):
   if i%1000==0: 
       print 'location %d out of %d'%(i,train_array.shape[0])
   #L2 distance of location
   loc_mat[i,:]=np.linalg.norm(np.subtract(train_array[i,0:2],
                                           train_array[:,0:2]), axis=1)
    
#Sort edges for each location in increasing order of euclidean distance
print 'sorting location matarix....'
sorted_loc=loc_mat.argsort(axis=1)
k=60#Neighbors of node to connect edges 6 ims are still in same lat,lng

sorted_loc_mat=np.zeros((loc_mat.shape[0],k))
for i in xrange(sorted_loc_mat.shape[0]):
    sorted_loc_mat[i,:]=loc_mat[i,sorted_loc[i,0:k]]

feat_array=np.asarray(train_feats[0:len(train_data)],dtype=np.float)
'''
#Feature difference matrix between each point and its K'th neighbors
feat_mat=np.zeros((train_array.shape[0],k))
for i in xrange(feat_array.shape[0]):
    if i%1000==0: 
        print 'features %d out of %d'%(i,feat_array.shape[0])
    #L2 distance of features
    feat_mat[i,:]=np.linalg.norm(np.subtract(feat_array[i,:],
                    feat_array[sorted_loc[i,0:k],:]),axis=1)  
'''


# In[ ]:

#for label=0 (lowest housing $$)
if USE_SVM:
    labels_0=np.copy(train_array[:,2]) #actual label
    labels_1=np.copy(train_array[:,2]) #actual label

    labels_0[np.where(train_array[:,2]==0)[0]]=1
    labels_1[np.where(train_array[:,2]!=0)[0]]=-1

    #Unary energies -ln(phi1)=Y_i*W*X_i
    unary_energy_0=np.multiply(svm_labels_0,
                      np.dot(np.asarray(w_0),np.transpose(feat_array)))

    unary_energy_1=np.multiply(svm_labels_1,
                      np.dot(np.asarray(w_0),np.transpose(feat_array)))
    
else:
    #Make lables into binary classes 0,1 and 2,3
    feat_array=np.asarray(train_feats[0:len(train_data)],dtype=np.float)
    probs_1=feat_array[:,0]+feat_array[:,1] #-->classes 0&1 are labeled 1
    probs_0=1-probs_1                       #-->classes 2,3 are labeled 0
    unary_energy_0= -np.log(probs_1)
    unary_energy_1= -np.log(probs_0)
    

    
actual_labels=train_array[:,2]
actual_bin_labels=(actual_labels==0)+(actual_labels==1)
cnn_bin_labels=np.argmax(np.array([probs_0, probs_1]),axis=0)
#Create binary energies
USE_VICENTE=False
max_acc=0
res_list=[]

if USE_VICENTE:
    #Alpha1 and Alpha2 from Vincente's paper
    alpha1=1e-3
    alpha2=1e-3
    binary_energy=np.add(np.divide(alpha1,sorted_loc_mat),
                           np.divide(alpha2,feat_mat))

    binary_energy[np.where(np.isinf(binary_energy))]=np.finfo('float').max

else:
    w1_list=[0,0.5,1,5,10,50,100,1000]
    w2_list=[0,0.5,1,5,10,50,100,1000] 
    ebs_list=[0,0.1,0.5,1,5,10]
    for w1 in w1_list:
        for w2 in w2_list:
            for ebs in ebs_list:
                print 'trying w1=%d and w2=%d and ebs=%d'%(w1,w2,ebs)
                binary_energy=w1*(ebs+np.exp(-w2*sorted_loc_mat))/(ebs+1)

                #Write these out to file for c program to read

                #Write (xi,xj,cost_ij)
                binary_energy_file='/imagenetdb3/tgebru/cvpr2016/binary_energies.txt'
                print 'Writing binary energies to file...'
                with open(binary_energy_file, 'wb') as f:
                    for i in xrange(binary_energy.shape[0]):
                        for j in xrange(k):
                            f.write('%d,%d,%f\n'%(i,j,binary_energy[i,j]))

                #Write (xi, cost_source_xi, cost_xi_sink)
                unary_energy_file='/imagenetdb3/tgebru/cvpr2016/unary_energies.txt'
                print 'Writing unary energies to file...'
                with open(unary_energy_file,'wb')as f:
                    for i in xrange(unary_energy_0.shape[0]):
                        f.write('%d,%f,%f\n'%(i,unary_energy_1[i],unary_energy_0[i]))

                #Run maxflow/mincut for different parameters
                import os
                flow_res_file='/imagenetdb3/tgebru/cvpr2016/flow_results.txt'
                code_dir='/afs/cs.stanford.edu/u/tgebru/cvpr2016/code/maxflow/'
                os.system('g++ -o %s/tg_out %s/graph.cpp %s/maxflow.cpp %s/tg_get_mincut.cpp'%(code_dir,code_dir,code_dir,code_dir))
                os.system('%s'%os.path.join(code_dir,'./tg_out'))

                node_labels=open(flow_res_file,'rb').readlines()
                predicted_labels=np.asarray([n.split(',')[-1].strip() 
                                             for n in node_labels],dtype='int')

                with open('/imagenetdb3/tgebru/cvpr2016/housing_train_2013_class_cnn.pickle', 'rb') as f:
                    cnn_labels=pickle.load(f)

                cnn_labels=np.asarray(cnn_labels)
                #print np.where(actual_labels==cnn_labels)[0].shape[0]/float(cnn_labels.shape[0])
                #print np.where(actual_labels==predicted_labels)[0].shape[0]/float(cnn_labels.shape[0])
                print np.where(cnn_bin_labels==predicted_labels)[0].shape[0]/float(cnn_bin_labels.shape[0])
                print np.where(actual_bin_labels==cnn_bin_labels)[0].shape[0]/float(cnn_bin_labels.shape[0])
                acc= np.where(actual_bin_labels==predicted_labels)[0].shape[0]/float(actual_bin_labels.shape[0])
                if acc>max_acc:
                    max_acc=acc
                    max_w1=w1
                    max_w2=w2
                    max_ebs=ebs
                res_list.append((w1,w2,ebs,acc))
    with open('crf_params.p','w') as f:
       pickle.dump(res_list,max_acc,max_w1,max_w1,max_ebs,f)
# In[ ]:



