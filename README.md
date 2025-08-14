# Star classification web application
This is an interactive application that allows users to perform stellar classification. It is based on dataset which consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey). Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.

App is made using following parameters from _star_classification_ dataset:
1. `u` = ultraviolet filter in the photometric system
2. `g` = green filter in the photometric system
3. `r` = red filter in the photometric system
4. `i` = near Infrared filter in the photometric system
5. `z` = infrared filter in the photometric system
6. `redshift` = redshift value based on the increase in wavelength
7. `class` = object class (galaxy, star or quasar object).

Application is divided into three parts:
1. Introduction - contains a description of the project (in Polish)
2. Exploratory data analysis - includes several plots used for data exploration (correlation matrix, PC1/PC2, histplot, boxplots)
3. Prediction model - this part allows users to set values for each parameter to get a prediction of the object (model uses LogisticRegression).

Dataset - https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17







