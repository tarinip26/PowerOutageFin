**Detects power outage utilising (user generated) features of timestamp, location(mangalore), temperature(highest in the day), humidity(lowest in the day), wind_speed(average), precipitation(lowest in the day), power_outage using LSTM**


----------------------------------------------------------------------------------------------------------------------------------------------------


**LSTM.ipynb**

Main Model to train data and evaluate test data

Loss vs Epoch

ROC AUC

Confusion Matrix + F1 Measure

Precision-Recall


----------------------------------------------------------------------------------------------------------------------------------------------------


**Power_outage.py**

Dataset geenrated by python code

All features (except power_outage and timestamp) trignometric function of timestamp and other feature

Power outage decided by extreme conditions occuring at certain timings

CSV saved as multivariate_timeseries_data_mangalore_extreme.csv


----------------------------------------------------------------------------------------------------------------------------------------------------


**WeightedBinaryCrossentropy.py**

In the model, due to increased possibility of non power outage (0), there is a heavy class imbalance of 0s to 1s as present in real world data.

Tensorflow code to modify weights in binary cross entropy, function used during model compilation

Array[0] - Increase => Prevent 1s appearing instead of 0s => Reduces chances of False Positives

Array[1] - Increase => Prevent 0s appearing instead of 1s => Reduces chances of False Negatives


----------------------------------------------------------------------------------------------------------------------------------------------------


mu














