# Scientific-Document-Generation-LSTM
We use a two layer LSTM to generate text by predicting one character at a time. The model is trained on a dataset of arxiv abstracts but it can easily be optimized for a different dataset.

The architecture is still not optimized but works ok with slight overfit. You can modify the network architecture to better fit your data.

## Usage:

1) In order to train your own model you must prepare your data set using the data_prep.py script. 

2) After the data are processed they should be positioned inside the /data folder. 

3) You can now run the doc_maker.py script that will build and train the model.

One pass over all training batches takes approximately 20 seconds in an nvidia 1060 gtx. In order to start getting good outputs you should train for at least 500 epochs.

## Dependencies:

1) numpy

2) pandas

3) csv

4) tensorflow

5) keras
