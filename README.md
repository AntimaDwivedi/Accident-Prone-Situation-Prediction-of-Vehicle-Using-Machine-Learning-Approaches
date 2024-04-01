# Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches
#### An increase in the number of deaths due to road accidents every year sheerly due to negligence or emotional behavior of drivers has become a major problem in India. People driving late at night, especially truck drivers, due to reasons such as fatigue, lethargy, long work hours, and sleepiness, often lose control of the road and cause accidents that may or may not be fatal to the passengers but surely result in loss. The best way to avoid these kinds of fatal circumstances should be to avoid driving at all when not feeling up to the task. ####

Thus, the project proposes a model that can warn the driver whether the car cabin is prone to accidents or
not. This will be achieved by measuring various characteristics of the moving car such as:<br>
##### <b>[1]</b> Drowsiness state through the eyes (percentage of eye closure)<br>
##### <b>[2]</b> Drowsiness state through the mouth (yawning) of the driver.<br>
##### <b>[3]</b> Pulse detection of the driver to gauge his/her stress level.<br>

The  objective of the project is to predict the possibility of an accident based on these characteristics according to
their respective thresholds in the form of discrete values. We have collected **more than 2000 instances** of data from
the real-time observation and checked the accident-prone probability on that dataset with the help of a machine
learning algorithm.<br><br>

#### The accuracy of machine learning models, based on these values shall help us in evaluating the features that must be present in an accident prevention system inside a car cabin. ####

##  Data Procurement and Processing  ##
Owing to the novelty of our work, limited datasets were available in the field of accident detection using researched 
features, so we extracted the features that will be focussed in this study by using certain programmed detection 
techniques. Over 2,000 instances of all such features were calculated by simulating an automobile cabin 
environment and obtaining the value of all features for 4 different drivers, 1 male, and 4 female test subjects. All 
data collection and processing procedures described in this work were implemented in Python. 
These features are stated as follows: 
1. **EYE ASPECT RATIO:** Contains the values of the percentage of eye-opened by the driver. 
2. **MOUTH OPEN RATIO:** Contains the values of mouth opened and duration of mouth opening of the 
driver. 
3. **PULSE_VAL:** Contains the pulse value of the driver. <br>

**Apart from the values stated above, the dataset also contains the following values of the following features:** <br>
#### (1) EAR_THRESHOLD: ####
Contains the values of whether or not the measured eye aspect ratio of the driver has crossed the threshold (0 for negative and 1 for positive). 
Eye aspect ratio is an important factor in detecting the drowsiness of a driver through his or her face detection. Eye 
aspect ratio is calculated by using the landmark model of machine learning. When the subject’s eye is open, the value 
of EAR is fixed but as soon as the subject’s eye is closed, the value drops to 0. In this way, the aspect ratio is 
useful for detecting drowsiness or sleepiness of the driver. There is a specific formula that we are using to calculate the eye-aspect ratio that is given below: <br>
 
       EAR = |p1-p5|+|p2-p4| / 2|p0-p3| 
![Screenshot 2024-03-27 150437](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/86386ffa-3f70-47cf-b27d-4c7cae94b4f4)
![Screenshot 2024-03-27 150446](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/d7817eb6-f3d2-4327-af02-c3d446312496)

**where, p1, p2, p3, p4, p5, p6 are landmark points on drivers face detected by camera.** <br>

If the value of **‘EYE ASPECT RATIO’** is greater than 0.35, then the model will consider the driver as in an active state, and the value of ‘EAR_THRESHOLD’ remains zero, and as soon as
If the value goes less than 0.35 then the model will consider the driver to become inactive and the value of 'EAR_THRESHOLD’ will turn out to be 1.
#### (2) YAWN_THRESHOLD: ####
Contains the values indicating whether or not the driver is yawning.
Yawn is a sign of getting drowsy. If a person is getting sleepy, there is a high possibility of getting a yawn.
That is why we are using this factor in our dataset to train our model to detect the accurate situation of the driver.
Yawn can be detected by using the same method of detecting the eye aspect ratio which is landmarks detection.
Yawn will be detected by calculating **MAR which is Mouth Aspect Ratio**. After predicting the landmarks, only the
mouth landmarks are required to calculate the Mouth Aspect Ratio (MAR) to predict if the driver is drowsy or not. <br>

       Mouth Aspect Ratio = |p1-p2| 
We are calculating the Mouth aspect ratio (MAR) by measuring the distance between the upper lip and lower lips, as shown
in the Figures.<br>
![Screenshot 2024-03-27 150413](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/3219c17f-6a81-4515-9b64-4bcac20334f5)
![Screenshot 2024-03-27 150424](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/25d41c15-1d53-4731-af3b-33090c0b67d9)


**Illustration of points estimation:** <br> 
(a) Distance of points for an open mouth, and <br>
(b) Distance of points for a closed mouth <br>
Here if the distance between the upper and lower lips is less than 20 the value of ‘YAWN_THRESHOLD’ will remain zero
and if the distance between the upper lip and the lower lip is greater than 20 then the value of ‘YAWN_THRESHOLD’
turns out to be 1.
#### (3) PULSE_THRESHOLD: #### 
Contains the values indicating whether or not the driver's pulse is unsuitable (0 for
negative and 1 for positive).
Photoplethysmography (PPG) is a technique that  can provide a means of heart rate measurement by
detecting **blood volume pulse (BVP)** in the human face. Heart rate means the number of beats per minute. We use
_**Haar feature**_ and _**cascade classifier**_ to detect the heart rate of the driver from face recognition. The ideal  threshold value is 12 bpm. We have applied a threshold value for the heart rate range if the value of
heart rate is less than 90 then the value of ‘PULSE_VAL’ remains 0 and if the value of heart rate is greater than
90 then the ‘PULSE_VAL’ value will become 1.

## Calculating target variable ##
The target variable of our dataset, i.e., **‘RESULT_ACCIDENT’** stores the possibility of accident occurrence given
a certain value of the forested features. The value of the target variable is calculated with the help of the following
algorithm:
1. Observe the value of EAR_THRESHOLD and compare it with the real-time EAR. <br>
2. Observe the value of PULSE_THRESHOLD and compare it with the real-time PUlSE_VAL <br>
3. Observe the value of YAWN_THRESHOLD and compare it with the real-time MOR.<br>
4. Note down the above three compared values in terms of 0 and 1 to calculate our target variable. <br>
5. Lastly, RESULT_ACCIDENT=1 if any of the values observed above result in 1, otherwise 0.

##  Prediction Module ##
With the help of the Scikit Learn library, we were able to fit the prediction models into our dataset. We executed a binary
classification task, where the featured values result in an accident if the threshold values of the features are exceeded
according to the algorithm, otherwise, it results in no accident.
In this paper, we tested 4 machine-learning models, namely <br>
**Logistic Regression,** <br>
**Support Vector Machine,** <br>
**KNearest Neighbours and,** <br>
**Naïve Bayes.** <br>
We compared the accuracy and F1 Scores of all these models as fitted on
the dataset to obtain the model with the highest accuracy on our dataset. <br>
**75% of the dataset values were used for training while remaining of the dataset values were used for testing.**
<br>

## Algorithms ##
We have collected more than 2000 instances of data from real-time observation and checked the accidents-prone probability on that dataset with the help of a machine learning algorithm. <br>

![Screenshot 2024-03-27 150313](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/cb6153bb-17d2-445e-9041-ade20078b211)
<br>

## Experiments and Results  ##
For the prediction experiments, each machine learning model computed the accuracy and F1 score of the dataset with  the help of a confusion matrix. The best-obtained result (Accuracy=0.98) is 13 percent points higher than the K-Nearest  Neighbour algorithm. The results can be observed from Table.<br>

Logistic Regression is very close to nearly  perfect, as can be seen from the table. <br>
![Screenshot 2024-03-27 150342](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/7f91ecd5-8672-4e1b-b004-88561fdb09ac)

## Conclusions ##
The proposed method works on stating many values according to the researched count of factors that are greatly responsible for car accidents all over the world, especially in India. With the help of the dataset created,  we built a formula that helps us establish the discrete binary values of the possibility of an accident. This  novel dataset has been very helpful in predicting the accuracy of the approach that we began working with, i.e.,  the factors that contribute to road accidents. According to previous data available as well as the results obtained  from our predictions, we successfully established the relationship between the factors that we started with and the  target variable.



