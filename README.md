# 𝑾𝑨𝑳𝑲 𝑹𝑼𝑵 𝑪𝑳𝑨𝑺𝑺𝑰𝑭𝑰𝑪𝑨𝑻𝑰𝑶𝑵
![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/c81c93a0-1d80-4cee-a09f-2a1021d4216f)
# **BUSINESS CASE**
**To Create a predictive model to classify whether a person is running or walking based on the given predictor variables**.

_**IMPORTING ALL THE NECESSARY LIBRARIES**_
_**LOAD THE DATA**_

## **DOMAIN ANALYSIS**
_Domain analysis is a critical step in any project, including a walking vs. running classification project based on the
provided data attributes. It involves gaining a deep understanding of the domain in which the project operates._ 

_**Domain Description**_:

-  **Domain**: _Human motion analysis and activity recognition using wearable sensors._  

-  **Context**: _The project aims to classify human activities (walking and running) based on sensor data collected 
    from wearable devices._    
    
-  **Key Stakeholders**: _Researchers, data scientists, fitness app developers, wearable device manufacturers._    

    
  _**Data Attributes**_:  

-  1) **Date and Time**: _These attributes provide the timestamp for each recorded data point. They may be useful for 
    analyzing patterns over time and establishing when the activities occurred._  

-  2) **Username**: _Indicates the user associated with the data. It could be relevant for user-specific analysis or 
    personalized activity recommendations._  

-  3) **Wrist**: _Specifies the wrist (left or right) where the wearable sensor was placed. This attribute can impact 
    the sensor data due to differences in motion patterns between wrists._  

-  4) **Activity**:_The target variable, representing the activity being performed (walking or running). This is the 
   label you're trying to predict._

-  5) **Acceleration (X, Y, Z)**: _These attributes provide acceleration data in three axes (X, Y, and Z). They are 
    crucial for understanding the movement patterns of walking and running._  

-  6) **Gyroscope (X, Y, Z)**: _Gyroscope data captures angular velocity around the X, Y, and Z axes. It can help in 
    identifying the rotational aspects of activities._

_**BASIC CHECKS**_
* head,tail,sample,info
* data.describe()
  - . _"count" prints the number of rows excluding null values. As all of the above features have their count values the same as total rows, there are no null values._
- . _"wrist" and "activity" are nominal features._
- . _"wrist" refers to the hand on which the device was worn while recording, it can take only two values i.e., 0 for "left" and 1 for "right"._  
- . _"activity" refers to the physical activity being performed during recording, 0 for "walk" and 1 for "run"._
- . _For binary variables, "mean" can give valuable information on skewness. Mean values of "wrist" and "activity" are roughly around 0.5 indicating the sample collection is not heavily skewed towards one of the values._  
- . _The remaining six features are (x,y,z) acceleration & orientation values measured by the device, and they are ratio features._  
- . _Percentile & mean values provide a decent understanding of the skewness for ratio features. If mean is closer to 25th or 75th percentiles more than 50th percentile, that indicates an underlying skewness in the distribution._  
- . _Quick glance tells us that "acceletation_y", "acceleration_z" have skewness in their distribution.
- 
  # EXPLORATORY DATA ANALYSIS:
  
  ## _IN EDA we basically understand our data by plotting graphs and analyze them on that basis which gives us a broad view of data and we can check outlier deviation and all the neccesary info for upcoming procedures so it is one of the important part of process we basically do three types of analysis as:_

1). Univariate analysis

2). Bivariate Analysis and

3)Multivariate analysis

* Univariate Analysis: In univariate analysis we get the unique labels of categorical features, as well as get the range & density of numbers.

*  Bivariate Analysis: In bivariate analysis we check the feature relationship with target veriable.

*  Multivariate Analysis: In multivariate Analysis check the relationship between two veriable with respect to the target veriable.

* Library Used: Matplotlib & Seaborn
* 
![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/d21169e3-90dd-4815-880b-2f1269e177b1)

_**INSIGHTS FROM UNIVARIENT ANALYSIS**_

_In this Blue represents the walking acitivity and orange represents the running activity based on the persons data_  

- **wrist**:- from the plotting we conclude there is skewness present the this data and at the middle there is no value and the data is skewed to the left side .  
- **acceleration_x**:- _from the graphical representation we can sya that orange graph i.e for running activity of a person is normally distributed where as walking is not and also 0.62 % data is left side skewed _  
- **acceleration_y**:- _from this graph it not properly distributed and continuosly overlapping and data is Right side skewed  0.91 value._  
- **acceleration_z**_:- _from this graph we can say that the graph is left side skewed_  
- **gyro_x**_:- _data is overlapping but we can say that it is not skwed.  
- **gyro_y**_:- _data is left side skewde.and minimum outliers are present.
- **gyro_z**_:- _data is having some outliers but evenly distributed not skewed to any of aside_

  
_**BIVARIATE ANALYSIS**_  
![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/7a397af8-6938-4d44-a328-a46a638cc9bc)
_**INSIGHTS FROM BIVARIENT ANALYSIS**_
- _The horizontal line inside each box represents the median value of the data for the respective activity._  
- _The height of the box represents the IQR, which contains the middle 50% of the data. A taller box indicates higher variability in the data._  
- _The lines extending from the boxes (whiskers) show the range of the data, excluding outliers. Outliers, if present, are shown as individual points outside the whiskers._  
- _Each box plot is labeled with the feature name on the y-axis and "Activity" on the x-axis, making it clear which feature is being examined._   


![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/eb94dfb3-d8f3-46d2-bd9d-03a90f1ba956)
![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/6f38677f-b2a1-4ea6-8708-dcf7c7a87172)
_**INSIGHTS**_
- _Hours of day: Most of the samples were recorded between 2pm and 8pm with the highest count coming from 6pm. The dip in sample count at 5pm looks out of place and worth noting._
- _Days of week: Sunday dominates the sample count which could be due to it being no work day. Rest of the days have similar sample counts except for Wednesday which has zero._

![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/51e64de1-31c9-496b-9c2e-ed9de88dbd3d)
_**INSIGHTS**_

- _Here we have createed a temporary dataframe by replacing "wrist" column values (0 with "left" and 1 with "right") and "activity" column values (0 with "walk" and 1 with "run") as it helps make the charts more intuitive._
- _Distribution is roughly even for different "activity", "wrist" values, maybe a bit skewed towards right wrist but not by much._
- _The third chart above illustrates that for walk we have more samples for right wrist and vice-versa for run._

  _**DISTRIBUTION PLOTS OF ACCELEROMETER AND GYROSCOPE DATA**_
  ![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/87480ee7-b651-4696-a51b-d233061d71a8)
  _**INSIGHTS**_
  - _For x-axis, accelerometer data is roughly symmetric and the double peak pattern is because of two "wrist" values. Same behavior is noticed in gyroscope data._
- _For y-axis, gyroscope data has normal distribution with mean = 0. Accelerometer data on the other hand looks skewed, and has the most inconsistent distribution among all the ratio features._
- _For z-axis, gyroscope data looks symmetric. Accelerometer data is slightly skewed but not as much as y-axis data._

  _**VISUALIZING ACTIVITY SPLIT ACCELEROMETER AND GYROSCOPE DATA**_
  ![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/ef8b4191-2039-480e-b133-9ea3fcb0efac)

  _**INSIGHTS**_
  - _"acceleration_x", "acceleration_z" show clear differentiation between walking and running, with running yielding much higher(+ve, -ve based on the wrist) values._  
- _"acceleration_y" shows some separation but not as pronounced as the two other dimensions._  
- _"gyroscope" data on the other hand look quite similar for walking and running._
- _For predictive analysis, acceleration_x could be the most important feature because of it's data distribution quality and ability to differentiate "activity". Although acceleration_x, acceleration_z do show sepration, they suffer from some inconsistencies in data distribution which might hamper their prominence. It'll be interesting to see the effects of gyroscope data._
 ![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/764f47cd-d418-4d23-897c-da8118528faa)

_**Handling outliers in a project like a walking vs. running classification task is crucial because outliers can negatively impact the performance of your machine learning model. Becuase of this we will predict based on the original one only**_

### **CHECKING SKEWNESS**

wrist: Skewness = -0.09

acceleration_x: Skewness = -0.62

acceleration_y: Skewness = 0.91

acceleration_z: Skewness = -1.84

gyro_x: Skewness = 0.07

gyro_y: Skewness = -0.02

gyro_z: Skewness = 0.04
_**BEFORE CHECKING SKEWNESS**_
![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/53dd1ff6-c80d-4545-ba19-818aaef59276)

_**GRAPH AFTER TRANSFORMING SKEWNESS**_
![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/97db8069-e45e-4e03-83b1-f68c5b84d479)

### **SPLITTING DATA INTO DEPENDENT AND INDEPENDENT VARIABLES**

## **SCALING**

- _Scaling is a preprocessing technique used in data analysis and machine learning to standardize or normalize the range of numerical features. It ensures that different features have similar scales, preventing some features from dominating others during modeling._
- _Here we used standardscalar for our scaling purpose.

  ### **CHECKING CORRELATION**
![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/3e0709a2-1a0a-4b66-b9a1-4fe291a8220e)

## **MODEL CREATION**

### **LOGISTIC REGRESSION**

     precision    recall  f1-score   support

           0       0.83      0.93      0.88     11040
           1       0.93      0.81      0.86     11107

    accuracy                           0.87     22147
    macro avg       0.88      0.87      0.87     22147
    weighted avg       0.88      0.87      0.87     22147

    0.8699598139702894

   ![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/a5be4d09-ece6-47e5-8530-d3b5c8446132)

### **DECISION TREE**

   Accuracy: 0.99
   
  Confusion Matrix:
  
  [[10859   181]
  
  [  180 10927]]
  
   Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.98      0.98     11040
           1       0.98      0.98      0.98     11107

    accuracy                           0.98     22147
    macro avg       0.98      0.98      0.98     22147
    weighted avg       0.98      0.98      0.98     22147

### **RANDOM FOREST ALGORITHM**

Accuracy: 0.99

Confusion Matrix:

 [[10936   104]
 
 [   82 11025]]
 
Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99       11040
           1       0.99      0.99      0.99       11107

    accuracy                           0.99       22147
    macro avg       0.99     0.99      0.99       24147
    weighted avg    0.99    0.99       0.99       22147


### **SUPPORT VECTOR CLASSIFIER**

Accuracy: 0.99

Confusion Matrix:

 [[10677   363]
 
 [ 2438  8669]]
 
Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.97      0.88     11040
           1       0.96      0.78      0.86     11107

    accuracy                           0.87     22147
    macro avg       0.89      0.87      0.87     22147
    weighted avg       0.89      0.87      0.87     22147

    ### **XG BOOST ALGORITHM**
Accuracy: 0.99

Confusion Matrix:

 [[10961    79]
 
 [   79 11028]]
 
Classification Report:

               precision    recall  f1-score   support

           0       0.99      0.99      0.99     11040
           1       0.99      0.99      0.99     11107

    accuracy                           0.99     22147
    macro avg       0.99     0.99      0.99     22147
    weighted avg    0.99    0.99        0.99    22147

    ### **CNN MODEL**
   ** Accuracy for CNN: 0.9821455478668213**

  ![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/654f5556-cb25-4007-a552-690eb98c1fae)

  ![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/eee42d7e-058d-4ba5-bf97-0f310e2f9129)

  ![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/e65c3ced-654f-4a11-a33f-e297fff9ede9)

  ### RNN MODEL
  **Accuracy: 0.9850**

  
### MLPClassifier
  **Accuracy :0.9901115275206575**

![image](https://github.com/Tanwar-12/WALK-RUN-CLASSIFICATION./assets/110081008/d85f722a-3867-4b56-9b83-cde92a2bbb8c)

_**FINAL CONCLUSION**_


_**Logistic Regression(Log)**_: _This model achieved an accuracy score of 0.86. While it's a decent score, it may not be the best choice if higher accuracy is crucial for your task. You might want to consider more complex models if you need better performance._  

_**Decision Tree (Decision)**_: _With an accuracy score of 0.98, the Decision Tree model performs very well on your dataset. It's a strong candidate for your task, and its simplicity makes it easy to interpret._  

_**Random Forest (Random)**_: _The Random Forest model also scored 0.99 in accuracy, indicating excellent performance. Random Forest is an ensemble method based on Decision Trees and is known for its robustness and accuracy._  

_**Support Vector Machine (SVM)**_: _SVM achieved an accuracy score of 0.87, which is reasonable but not as high as the scores of the Decision Tree and Random Forest. SVM can be effective in some cases, but it might not be the best choice for your specific dataset._  

_**XGBoost (XGBoost)**_: _Similar to the Random Forest, XGBoost also scored 0.99 in accuracy. XGBoost is a powerful ensemble algorithm known for its high performance and is often a top choice in competitions and real-world applications._  

_**Convolutional Neural Network (CNN)**_: _The CNN model achieved an accuracy score of 0.98, which is quite impressive if you're working with image data. CNNs are specifically designed for image-related tasks and perform exceptionally well in these domains._  

_**Recurrent Neural Network (RNN)**_: _Similar to XGBoost and Random Forest, RNN scored 0.99 in accuracy. RNNs are well-suited for sequential data and time series tasks and seem to be excelling in your dataset._  

_**These scores suggest that the **Random Forest**, **XGBoost**, and **RNN models** have the highest accuracy among all the models**_


_**NOW TO GET MORE ACCURATE MODEL FOR OUR PEOBLEM STATEMENT WE HAVE DONE HYPERPARAMETER TUNNING ON BOTH THE MODEL OF **Random Forest**, **XGBoost** AND THERE IS SLIGHTLY CHANGE OF MINIMUM POINTS AS THE VALUE IS DECREASED SO BECUASE OF THIS  WE CAN CONCLUDE FROM THAT**_

_**THE **RNN** IS THE BEST SUITED MODEL FOR OUR WALK RUN CLASSIFICATION PROJECT**_ 
