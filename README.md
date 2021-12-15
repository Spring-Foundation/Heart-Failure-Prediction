# Heart Failure Prediction and Survival Analysis

- Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
- Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

- Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

- People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

## Idea and Application

- Our team is developing a powerful machine Learning model to identify the patients at risk of death following a heart attack. 
- This model will be very useful when deployed at hospitals since at the time of discharging a patient, it will predict the likelihood of death for the patient as well as the survival period. 
- This can be used by hospitals to prioritise high risk cases and they can be monitored more closely and follow-up care can be provided to this risky group of patients’ post-discharge.

## Solution
**A.	Predicting the Likelihood of Death of the Patient**

The prediction of the probability of the death will based on the different features mentioned above. Firstly, we will clean the data and add feature engineering such as label encoding, temporal substitution, sampling, and feature selection based on their importance. Later the pre-processed data is passed on to different models for training. Based on their result and evaluating based on the accuracy and recall values the best model will be finalized. 

**B.	Survival Probability of the Patient for ‘n’ period (time)**

The second solution we are presenting is about the survival chances of a patient on the nth observed day. Within the dataset we have a feature described as ‘time’ which states the follow-up period of the patient. The time feature varies from 4(min) to 275(max). Thus, the survival model created based on a few important features will provide the hazard score and survival score for the patients.

Hazard score indicates the risk of death due to heart failure. Higher the hazard score, more the risk of getting a heart complication. Another parameter we mentioned is survival score. Survival Function calculates survival probabilities of the patient over the course of time. The value of survival score lies between 0 to 1. These results help the doctors to prioritize the patients and make improvements in their medicines so as to increase the survival rate of the patients.
