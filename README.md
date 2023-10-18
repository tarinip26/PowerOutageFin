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


**multivariate_timeseries_data_mangalore_extreme.csv**

User generated Comma Separated Data with the features as mentioned above


----------------------------------------------------------------------------------------------------------------------------------------------------


**po-model.data-00000-of-00001 & po_model.index**

Best current pretrained weights saved

Use model.load_weights to load the weights


----------------------------------------------------------------------------------------------------------------------------------------------------

```
Sequential Model:

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 bidirectional (Bidirectiona  (None, 16, 256)          136192    
 l)                                                              
                                                                 
 re_lu (ReLU)                (None, 16, 256)           0         
                                                                 
 dropout (Dropout)           (None, 16, 256)           0         
                                                                 
 bidirectional_1 (Bidirectio  (None, 16, 128)          164352    
 nal)                                                            
                                                                 
 re_lu_1 (ReLU)              (None, 16, 128)           0         
                                                                 
 dropout_1 (Dropout)         (None, 16, 128)           0         
                                                                 
 bidirectional_2 (Bidirectio  (None, 16, 128)          98816     
 nal)                                                            
                                                                 
 re_lu_2 (ReLU)              (None, 16, 128)           0         
                                                                 
 dropout_2 (Dropout)         (None, 16, 128)           0         
                                                                 
 lstm_3 (LSTM)               (None, 128)               131584    
                                                                 
 dropout_3 (Dropout)         (None, 128)               0         
                                                                 
 dense (Dense)               (None, 1)                 129       
                                                                 
=================================================================
Total params: 531,073
Trainable params: 531,073
Non-trainable params: 0
_________________________________________________________________
```


F1 measure: 0.9

ROC AUC: 0.9364620938628158



![image](https://github.com/speedwagon1299/PowerOutage/assets/118172807/7c61ec58-685d-41f4-a221-65c5989f4758)


![image](https://github.com/speedwagon1299/PowerOutage/assets/118172807/51c0d8de-ff9d-43f1-93a6-09a15c0ced2b)


![image](https://github.com/speedwagon1299/PowerOutage/assets/118172807/ec4d6583-8249-4021-a47d-fa196458d732)
