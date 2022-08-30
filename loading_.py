from fileinput import filename
from scipy.io.matlab.mio import loadmat
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
#import vgg_19
import numpy as np
import cv2
import tables
import scipy.io
import scipy.io as spio
import scipy.io as si
import h5py
import glob
import os
import time
import matplotlib.pyplot as pyplot
from tensorflow.python import pywrap_tensorflow
import inspect
'''
def Load_Dataset():
   # p = h5py.File('lfw_att_73.mat')
    address = []
    address_train=[]
    address_test=[]
    label_train=[]
    label_test=[]
    f = h5py.File('address.mat')

    for column in f['Add']:
       row_data = []

       for row_number in range(len(column)):
           row_data.append(""+''.join(map(unichr, f[column[row_number]][:]))+"")
       address.append(row_data)

  #     T="" +row_data[0]+ ""
 #      print T
    count=0
    i=0

    set_variable=True
    label = f['labels']
    Label=np.transpose(label)
    Len=(len(Label))
    class_number=np.max(Label)+1
    train_index=[]
    test_index=[]
    _Sample=np.zeros(np.int(class_number))
    while i<Len:
         if  count<class_number and _Sample[count]<1:
             _Sample[count]=_Sample[count]+1
             test_index.append(i)

             address_test.append(address[i])
             label_test.append(Label[i])
             i=i+1
         else:
             while i<Len and Label[i]==count:
               train_index.append(i)

               address_train.append(address[i])

               label_train.append(Label[i])
               i=i+1
             count = count + 1
  #  print 'salam'
    #Address=np.transpose(address)

    return address_train,label_train,address_test,label_test

     #  return  address_test,np.transpose(label_test)
'''
def Load_Dataset_train():
    address1 = []
    f = h5py.File('TF_Final/670nm10/670nm10_training.mat')
    for column in f['name_train']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append("" + ''.join(map(unichr, f[column[row_number]][:])) + "")
            if os.path.exists(row_data[0]):
             address1.append(row_data[0])
            row_data = []
    label = f['label_train']
    Label = np.transpose(label)
    return address1, np.int32(label)
def Load_Dataset_test():
    address1 = []
    f = h5py.File('TF_Final/670nm10/670nm10_testing.mat')
    for column in f['name_test']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append("" + ''.join(map(unichr, f[column[row_number]][:])) + "")
            address1.append(row_data[0])
            row_data = []
    label = f['label_test']
    Label = np.transpose(label)
    return address1, label