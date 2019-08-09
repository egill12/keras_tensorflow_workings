# keras_tensorflow_workings
Code workings in Keras and Tensor flow to use various Machine Learning models for currency trading.
The code is kept in its own separate envvironment as this is the 64 bit version. The majority of the code here will be used to implement ANNs and more specifically Recurrent Neural Networks (LSTM) There is also a section which is dedicated to running the code on a cloud server in order to speed up the training process.
The file collects the normalised versions of each curreny data, using meaures of price based 
data only.
This NN file uses a LSTM (the lookback again is of interest, do we need to vary these?)
Should we standardise the time zone changes? Apply entropy on the model returns? 
Test on randomly generated data, use the valuation, eco trends, risk aversion etc to 
see if there is any interesting patterns there which allow us to understand the model behaviour
