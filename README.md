# Lithium-ion-RUL Prediction

Our suggested method uses an Artificial Neural Networks (ANN) model to forecast how long a lithium ion battery will be functional. An Artificial Neural Network(ANN) model using a layered structure, consisting of multiple nodes in  every layer. An activation function is used to carry out operations at corresponding layers. The implementation  of our proposed solution begins with collecting the data to be worked upon. Python is used to implement the following solution. NASA Ames Prognostics Center of Excellence’s dataset provides multiple datasets for research purposes, publicly. This data is stored in a Dataframe object. The data   is preprocessed by removing rows consisting of unavailable data and removing the ’SampleId’ attribute, as it  doesn’t  serve any purpose in our implementation. The dataset is then normalized using mean and standard deviation. This helps us in achieving faster convergence.
We've also compared our primary model against CNN, Linear Regression and Support Vector Machine(SVM) models.
