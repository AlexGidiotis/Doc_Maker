#============================================================ Imports =======================================================
import numpy as np
import random
import time
import re
import csv
import pandas as pd
import os

#========================================================= Global variables =================================================
# Modify this path
root_path = '/home/alex/Documents/Data/arxiv_data/'
test_split = 0.1

#=========================================================== Definitions ====================================================
def read_data():
	# Read all the data.
	df = pd.DataFrame()
	for doc in sorted(os.listdir(root_path)):
		if doc.split('_')[1] != 'dump': continue
		df_temp = pd.read_csv(root_path+doc, usecols=['abstract', 'categories'])
		df = df.append(df_temp, ignore_index=True)
	# Shuffle the dataset.
	df = df.sample(frac=1).reset_index(drop=True)


	# Split to train and test set.
	train_df = df[:int((1-test_split)*len(df))].reset_index(drop=True)
	test_df = df[int((1-test_split)*len(df)):].reset_index(drop=True)
	print train_df.shape[0],'training examples'
	print test_df.shape[0],'test examples'

	return train_df,test_df

def write_data(X_train,X_test):
	# Write the outputs to .csv
	print 'Writting...'
	with open("data/train_set.csv", "wb") as f:
	    for x in X_train:
	    	f.write('%s\n' %x)
	with open("data/test_set.csv", "wb") as f:
	    for x in X_test:
	    	f.write('%s\n' %x)

#========================================================== Main function ===================================================


train_df, test_df = read_data()

# Preprocess the data and labels for the train and test set.
X_train = []
y_train = []
for c,abstr in enumerate(train_df['abstract'].tolist()):
	X_train.append(abstr)
	if c % 10000 == 0: print c
X_test = []
y_test = []
for c,abstr in enumerate(test_df['abstract'].tolist()):
	X_test.append(abstr)
	if c % 10000 == 0: print c

write_data(X_train,X_test)
