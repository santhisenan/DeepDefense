# Using Bidirectional LSTM based RNN to classify DDoS attack packets

The design of this classifier was inspired by [DeepDefense: Identifying DDoS Attack via Deep Learning](https://ieeexplore.ieee.org/document/7946998) paper. 

The CSV extract of the dataset can be found [here](https://gitlab.com/santhisenan/ids_iscx_2012_dataset).

## Architecture of the model
![Model](architecture/model_brnn.png)


## Usage

Run the ```brnn_classifier.ipynb``` notebook.

## Results

### Plot of accuracy
![Plot of accuracy](results/BRNN_Model_Accuracy_40epochs.png)

### Plot of loss
![Plot of loss](results/BRNN_Model_Loss_40epochs.png)

