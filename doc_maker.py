#============================================================ Imports =======================================================
import numpy as np
import random
import time
import sys

from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm


#========================================================= Global variables =================================================
# Modify this paths as well
DATA_DIR = 'data/'
TRAIN_FILE = 'train_set.csv'
# Max number of words in each sequence.
MAX_SEQUENCE_LENGTH = 100
SKIP = 2
# The name of the model.
STAMP = 'doc_maker'

#=========================================================== Definitions ====================================================
class Corpus(object):
	# Data generator.
	# INitialize the input files.
	def __init__(self,in_file,target_file=None):
		self.in_file = in_file
		self.target_file = target_file
		self.__iter__()

	# Yield one row per iteration. Each row is an abstract (preprocessed)
	def __iter__(self):
		for i,line in enumerate(open(self.in_file)):
			yield line.strip()+' '



# Sample from the predictions using a random threshold. Return the one-hot encoded sample.
def sample(prediction,char_size):
	# Random uniform threshold
	r = random.uniform(0,1)
	s = 0
	char_id = len(prediction) - 1
	for i,pr in enumerate(prediction):
		s += pr 
		if s >= r:
			char_id = i
			break

	# One hot encode.
	char_one_hot = np.zeros((char_size))
	char_one_hot[char_id] = 1.

	return char_one_hot


def read_data():
	train_set = Corpus(DATA_DIR+TRAIN_FILE)

	# LOad the data.
	X_data = ''
	for c,vector in enumerate(train_set):  # load one vector into memory at a time
		X_data += vector
		if c > 30000: break
		if c % 10000 == 0: 
			print c


	print len(X_data), 'training characters'


	print 'Tokenizing text...'
	# Tokenize and pad text.
	chars = sorted(list(set(X_data)))
	char_size = len(chars)
	# Word index for word 2 id mapping.
	char2id = dict((c,i) for i,c in enumerate(chars))
	# Inverse word index for id 2 word mapping.
	id2char = dict((i,c) for i,c in enumerate(chars))
	print('Found %s unique tokens' % char_size)


	sections = []
	next_chars = []
	# We iterate over the data and split to sections.
	for i in range(0,len(X_data)-MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH):
		sections.append(X_data[i: i + MAX_SEQUENCE_LENGTH])
		next_chars.append(X_data[i + MAX_SEQUENCE_LENGTH])


	# Vectorize the data and labels.
	X_train = np.zeros((len(sections),MAX_SEQUENCE_LENGTH,char_size))
	y_train = np.zeros((len(sections),char_size))
	for i,section in enumerate(sections):
		for j,char in enumerate(section):
			X_train[i,j,char2id[char]] = 1.

		y_train[i,char2id[next_chars[i]]] = 1.

	print 'Shape of data tensor:', X_train.shape
	print 'Shape of label tensor:', y_train.shape

	return X_train, y_train, char2id, id2char


#========================================================== Main function ===================================================
X_train, y_train, char2id, id2char = read_data()
char_size = X_train.shape[2]

#========================================================== Build the net ===================================================
batch_size = 256
epochs = 2000
starting_text = 'computers are amazing'

input_shape = (X_train.shape[1],X_train.shape[2],)

input_layer = Input(shape=(input_shape), dtype='float32')

lstm1 = LSTM(512, activation='tanh', recurrent_activation='hard_sigmoid', recurrent_dropout=0.0, dropout=0.2, 
			kernel_initializer='glorot_uniform', kernel_constraint=maxnorm(3), return_sequences=True)(input_layer)

lstm2 = LSTM(512, activation='tanh', recurrent_activation='hard_sigmoid', recurrent_dropout=0.0, dropout=0.5, 
			kernel_initializer='glorot_uniform', kernel_constraint=maxnorm(3), return_sequences=False)(lstm1)

drop = Dropout(0.5)(lstm2)
dense = Dense(char_size, activation='softmax', kernel_initializer='glorot_uniform')(drop)


model = Model(input_layer,dense)
model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop')
model.summary()
print(STAMP)

#======================================================== Training =========================================================
# Save the model.
model_json = model.to_json()
with open(STAMP + ".json", "w") as json_file:
	json_file.write(model_json)


# The training process
for epoch in range(epochs):
	print 'Epoch:',epoch
	# Train for one iteration over all batches
	hist = model.fit(X_train, y_train, batch_size=batch_size, shuffle=False, validation_split=0.1, epochs=1)

	# Save every 10 epochs
	if epoch % 10 == 0:
		model.save_weights(STAMP+'.hdf5', True)
	# Generate text
	if epoch % 10 == 0:
		test_generated = ''
		test_generated += starting_text
		sys.stdout.write(test_generated)
		# Generate 400 characters.
		for i in range(400):
			# First vectorize the starting text.
			x = np.zeros((1, MAX_SEQUENCE_LENGTH, char_size))
			for t, char in enumerate(starting_text):
				x[0, t, char2id[char]] = 1.
			# Predict using the previously generated text as input.
			preds = model.predict(x, verbose=0)[0]
			# Sample and decode the prediction.
			next_char_one_hot = sample(preds,char_size)
			next_char = id2char[np.argmax(next_char_one_hot)]
			# Append prediction to the generated text.
			test_generated += next_char
			# Write the output one char at a time.
			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()