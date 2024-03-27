# Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches
Increase in the number of deaths due to road accidents every year sheerly due to negligence or emotional behavior
of drivers has become a major problem in India. People driving late at nights, especially truck drivers, due to
reasons such as fatigue, lethargy, long work hours and sleepiness, often lose control on the road and cause
accidents which may or may not be fatal to the passengers but surely results in loss. The best way to avoid these
kinds of fatal circumstances should be to avoid driving at all when not feeling up to the task.

Thus, the project proposes a model that is able to warn the driver whether the car cabin is prone to accidents or
not. This will be achieved by measuring various characteristics of the moving car such as:<br>
##### <b>[1]</b> Drowsiness state through the eyes (percentage of eye closure)<br>
##### <b>[2]</b> Drowsiness state through the mouth (yawning) of the driver.<br>
##### <b>[3]</b> Pulse detection of the driver to gauge his/her stress level.<br>

The main aim of the project is to predict the possibility of accident based on these characteristics according to
their respective thresholds in the form of discrete values. We have collected **more than 2000 instance** of data from
the real time observation and checked the accidents prone probability on that dataset with the help of machine
learning algorithm.<br><br><br>

The accuracy of machine learning models, based on these values shall help us in evaluating the features that must
be present in an accident prevention system inside a car cabin.

#### 3.1 Data procurement and Processing  ####
Owing to the novelty of our work, limited datasets were available in the field of accident detection using researched 
features, so we extracted the features that will be focussed in this study by using certain programmed detection 
techniques. Over 2,000 instances of all such features were calculated by simulating an automobile cabin 
environment and obtaining the value of all features for 4 different drivers, 1 male and 4 female test subjects. All 
data collection and processing procedures described in this work were implemented in Python by the authors. 
These features may be stated as follows: 
1. **EYE ASPECT RATIO:** Contains the values of percentage of eye-opened of the driver. 
2. **MOUTH OPEN RATIO:** Contains the values of mouth opened and duration of mouth opened of the 
driver. 
3. **PULSE_VAL:** Contains the pulse value of the driver. <br>

Apart from the values stated above, the dataset also contains the following values of the following features: 
##### (1) EAR_THRESHOLD: ##### 
Contains the values of whether or not the measured eye aspect ratio of the driver has crossed the threshold (0 for negative and 1 for positive). 
Eye aspect ratio is an important factor on detecting drowsiness of driver through his or her face detection. Eye 
aspect ratio is calculated by using landmark model of machine learning. When the subject’s eye is open, the value 
of EAR is fixed but as soon as the subject’s eye is closed, the value drops to 0. In this way eye aspect ratio is 
useful to detect drowsiness or sleepiness of driver. There is a specific formula that we are using to calculate eye 
aspect ratio that is given below: 
 
**EAR = |p1-p5|+|p2-p4| / 2|p0-p3|** <br>![Screenshot 2024-03-27 150446](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/d7817eb6-f3d2-4327-af02-c3d446312496)
![Screenshot 2024-03-27 150313](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/cb6153bb-17d2-445e-9041-ade20078b211)

where, p1, p2, p3, p4, p5, p6 are landmark points on drivers face detected by camera.![Screenshot 2024-03-27 150342](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/7f91ecd5-8672-4e1b-b004-88561fdb09ac)

![Screenshot 2024-03-27 150437](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/86386ffa-3f70-47cf-b27d-4c7cae94b4f4)

![Screenshot 2024-03-27 150424](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/25d41c15-1d53-4731-af3b-33090c0b67d9)


![Screenshot 2024-03-27 150413](https://github.com/AntimaDwivedi/Accident-Prone-Situation-Prediction-of-Vehicle-Using-Machine-Learning-Approaches/assets/56269029/3219c17f-6a81-4515-9b64-4bcac20334f5)
