# Scientific-Document-Generation-LSTM
We use a two layer LSTM to generate text by predicting one character at a time. The model is trained on a dataset of arxiv abstracts but it can easily be optimized for a different dataset. This is a toy project that illustrates how to use keras to build a text generating LSTM.

The architecture is still not optimized but works ok with slight overfit. You can modify the network architecture to better fit your data. If data set is too small (less than 1 million characters) you might want to use less LSTM units as the model tends to overfit. 

A good way to check for overfitting is to monitor the validation loss. If at some point the validation loss stops decreasing while the training loss is still decreasing, it means that the model might be overfitting.

## Usage:

1) In order to train your own model you must prepare your data set using the data_prep.py script. 

2) After the data are processed they should be positioned inside the /data folder. 

3) You can now run the doc_maker.py script that will build and train the model.

One pass over all training batches takes approximately 20 seconds in an nvidia 1060 gtx. In order to start getting good outputs you should train for at least 500 epochs.

** Be careful with the memory usage. ** Loading and tokenizing 5000 lines of text can easily take up to 5GB of memory. You can control the amount of lines by modifying the LINES_TO_READ parameter in the doc_maker script. 

## Dependencies:

1) numpy

2) pandas

3) csv

4) tensorflow

5) keras
