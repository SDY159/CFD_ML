# CFD_ML
#Predicting erosion rate with ML based on the CFD data

This project is about a highly accurate and efficient ML approach for particle trajectory and erosion rate prediction. Our model consists of two ML models. The first model predicts particle trajectories using the following initial conditions: particle size, main-inlet speed, sub-inlet speed, main-inlet pressure, and sub-inlet pressure. The second model predicts the erosion rate profile in the 3-D space using the particle trajectories obtained from the first model. We use the Long and Short-Term Memory networks (LSTM) for the first model, a specialized recurrent network adapted to work for time series data. LSTM is a state-of-the-art methodology for classifying, processing, and predicting values for a given time series. For our second model, we carry out the erosion rate prediction using a 3-D Convolutional Neural Networks (CNN) based on the output obtained from the LSTM model. 

A simplified part of the steam distribution header in the OP-650, a boiler model for sub-critical coal plants, was used for the geometric model for erosion rate prediction.
Operating parameters
Particle size: 40-56 micrometer
Main-inlet speed: 8.45-14.85 m/s
Sub-inlet speed: 25-57 m/s
Main-inlet pressure: 12.00-15.84 MPa
Sub-inlet pressure: 1.0-2.6 MPa

Total 3,125 datasets

This project is divided into 8 levels, according with the name of directories.

level_1/ : Ansys Fluent CFD data processing codes and journal files

level_2/ : LSTM for particle trajectory prediction

level_3/ : LSTM + CNN for Erosion rate prediction based on the output of LSTM

level_4/ : Codes for feature importance analyses

level_5/ : Hyperparameter optimization for CNN

level_6/ : Data visualization processing

