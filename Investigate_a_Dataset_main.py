#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a dataset: No-Show Appointments.
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# In this project, from the list of dataset provided, i have selected the No-show appointments dataset available on kaggle via this link: https://www.kaggle.com/joniarroba/noshowappointments/home . This dataset has been collected from hospitals in brazil showing patients that showed up for their appoints and those that didn't. I will be analysing this dataset to find reasons and correlation between patients that showed and those that didnt using other attributes i have been provided with.
# 
# | Column Name | Significance |
# | --- | --- |
# | PatientId | This is the Patient's identifier |
# | AppointmentID | Idnetifier for patients appointment |
# | Gender | sex of patient, could be male or female |
# | ScheduledDay | Day of scheduled appointement |
# | AppointmentDay | Day meant forthe patient's appointment |
# | Neighbourhood | Shows hospital's location |
# | Scholarship | Shows if the patient has some form of scholarship |
# | Hipertension | Shows if the patient is hypertensive or not |
# | Diabetes | Shows if the patient is diabetic or not |
# | Alcoholism | Shows if the patient is an alcoholic or not |
# | Handicap | shows if the patient has some form of disability |
# | SMS_received | Shows if the patients got a sms or didn't |
# | No-show | Tells if a patient showed up for thier appointment or didn't |
# 
# 
# 
# ### Question(s) for Analysis
# 1. I would like to analyse the correlation between the age of the patients. It would be great knowing how the different age groups show up for their appointments/Not.
# 
# 2. I would also analyse how gender plays a role in the patients meeting up with their appointments.
# 
# 3. I would also try to show a correlation if patients that received an sms reminder showep up more than the patients that didn't.
# 

# ### Data Wrangling

# In[1]:


# Setting up all the packages i intend using.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loaded my data and printed out a few lines below. 

df = pd.read_csv('noshowappointments2016.csv')
df.head(2)


# In[3]:


# Getting the dimensions of the dataframe
df.shape


# There are 110527 rows and 14 columns in the dataframe

# In[4]:


#lets get a general information about the dataframe
df.info()


# In[5]:


# Check for statistical data of the numnerical values.
df.describe()


# <b>The maximum value in the column name Handicap is 4 which in itself is an abnormally. The only reasonale values are 0 and 1. O means not handipped while 1 means hadicapped. Would probe further and clean accordingly.<b>

# In[6]:


# colums with values not equal to 0 and 1

# handicap_rem = df.loc[(df['handicap'] != 0) & (df['handicap'] != 1)]
# handcap_rem


# In[7]:


# check the tuple of the dataframe above's dimension.

# handcap_rem.shape


# <b>There are 199 rows of patients with invalid values. Values on the handicap column should be 0 and 1, but we got some irregularities like values 2,3 and 4. I would have to drop this column because it has way too much wrong values. I had to comment the above cell out as this generated an error(since the handicap column no longer exists) due to running the cells after completion. <b>

# In[8]:


##check number of duplicates

sum(df.duplicated())


# We also do not have duplicated values in our dataset.

# In[9]:


# Check for missing values
df.isnull().sum()


# This shows that there aren't missing values in our dataframe. This might not always be the case.

# In[10]:


#ploting an histogram to get a feel of my variables.

df.hist(figsize=(18,16));


# <b>Discussing the structure of the data and any problems that need to be cleaned</b>
# 
# <p>
#     1. I discovered there are no missing values<br>
#     2. The column names aren't all in standard naming conventions. I would make sure to     
#        correct that.<br>
#     3. I would also correct the the scheduledDay and appointmentDay columns type to daytime.<br>
#     4. I also discovered that the Age column has a minimum value of -1. In reality no one has an age of -1, so i might likely drop such rows.<br>
#     5. I would not be needing the Patient_id and the Appointment_id, I won't be dropping them off the table. Would just let them be.<br>
#     6. I would be replacing the strings Yes and No in the No_show column to 1 and 0 for easy analysis.<br>
#     7. In the handicap column, there is/are maximum values above 0 and 1. O is meant to show patients without disability and 1 for patients with disabilty, any other value is an outlier and should prodded further and cleaned if neccessary.<br>
#     8. From the histogram plotted above, i got a general overview of major variables in my dataset. Age is slighly skewed to the left, very few patients that are involved with alcoholism. Few patients daignosed with diabetes, e.t.c.
# </p>

# ### Data Cleaning

# <b> Standardizing all column names, due to non comformity amongst all column names<b>

# In[11]:


#checking all column names

df.columns


# In[12]:


# correct and standardize columns name

df.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day',
       'appointment_day', 'age', 'neighbourhood', 'scholarship', 'hypertension',
       'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']

# confirm if the changes have been made succesfully.

df.columns


# <b>Checking and correcting the datatypes of some colums</b>

# In[13]:


# change the datatype of the sheduled_day to datetime

df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])

#confirm the changes made
df.dtypes


# In[14]:


# change the datatype of the appointment_day to datetime

df['appointment_day'] = pd.to_datetime(df['appointment_day'])

#confirm the changes made
df.dtypes


# <b>Cleaning the age column with irregular values</b>

# In[15]:


# minimum age is -1 , lets see how many rows have age less than 0

df[df['age'] < 0]


# We do have 1 row with an age of -1. Since it is practically impossible to have such age, we would just drop such row.

# In[16]:


# drop row with age less than 0

df.drop(df[df['age'] < 0].index, inplace =True)

#lets confirm if the row has been dropped.

df[df['age'] < 0]


# In[17]:


# drop age greater than 100 as these are considered as ouliers

df[df['age'] > 100 ]

df.drop(df[df['age'] > 100].index, inplace =True)


# In[18]:


#lets confirm if the row has been dropped.

df[df['age'] > 100]


# ###### Though ages just above 100 are still very feasible, i would still consider them outliers and drop them off the dataset.

# <b># Dropping the handicap column, due to so much invalid values.<b>

# In[19]:


# drop the handicap column
df.drop('handicap', axis=1, inplace=True)

#confirm that the column has been dropped
df.head(2)


# In[20]:


df.describe()


# In[21]:


# Replacing the strings Yes and No in the No_show column to 1 and 0 respectively.
df = df.replace(to_replace = ['Yes','No'],value = ['1','0'])


# In[22]:


df.head(5)


# <b>Data cleaning session summary.<b>
# 
# 1. There where some incorrect datatypes which i had to deal with. This was to maintain accuracy and prevent data loss.
# 
# 2. The age column had some irregular values like -1. It is practically impossible for a patient to have such an age, so it has to be removed so as not to derail our analysis. I did same for ages above 100. Even though in real world scenerios, people age more than 100 but it became irregular for this dataset.
# 
# 3. The handicap column should have only two(2) values 0 and 1. O means not handicapped while 1 means the patient was handicapped. It was practically not possible to correct the values, so the column was dropped altogether.

# ### Exploratory Phase

# ### Research Question 1 : Could there be a relationship between the patient and their ability to show up for their scheduled appointments?

# In[23]:


df['age'].value_counts()


# In[24]:


# seperate the ages into proper groups for ease on visualization

age_grp = [0,20,40,60,80,100]
df['age_grp'] = pd.cut(df['age'],bins = age_grp)
age_grp_count = df.groupby('age_grp')['no_show'].value_counts()


# In[25]:


df['age_grp'].value_counts().sort_values(ascending = False)


# In[26]:


(df['age_grp'].value_counts().sort_values(ascending = False))/len(df)*100


# In[27]:


age_grp_count


# In[28]:


#bar chart showing our various age groups and how they showed up.
(age_grp_count/age_grp_count.groupby(level = 0).sum()*100).unstack().plot(kind = 'bar', figsize = (12,8))
plt.legend(['showed up','did not show up']);


# In[29]:


#Age distribution of patients:Who showed up and otherwise

showed_up = df['no_show'] == '0'
did_not_show_up = df['no_show'] == '1'
df[showed_up].age.hist(alpha = 0.75, bins = 20)
df[did_not_show_up].age.hist(alpha = 0.75, bins = 20)
plt.title('Age distribution of patients:Who showed up and otherwise.')
plt.legend(['Showed up','Did not show up'])
plt.ylabel('Total')
plt.xlabel('Age');


# <b>Note: In the dataset provided, it tells that No in the no_show column simply means the patient showed up for the scheduled appointment, while Yes means the patient never showed up.</b>

# Deductions from this visualizations shows that patients between the ages of 60-100 showed up for their appointments most. This can't be far fetched as they are aged and prone for all kind of illness. Could age be a factor for patients to show up more? I can't deduce that as i can see patients also within 100-120 didnt show up more often as they should considering my initial deductions.

# ### Research Question 2 : Do the Male gender show up more than the female gender also could there be underlining reasons why?

# In[30]:


df['gender'].value_counts()


# In[31]:


# Gender Plot

sns.countplot(df['gender']);
plt.xlabel('Gender')
plt.ylabel('Total')
plt.title('Gender Bar Plot');


# Note: F = Female while M equals Male. Therefore there are more female patients than male.

# In[32]:


df_sex = df.groupby('gender')['no_show'].value_counts().unstack()
df_sex.plot(kind = 'bar',figsize= (8,4))
plt.ylabel('Total')
plt.xlabel('Sex')
plt.title('Sex of gender based on no-show or show')
plt.legend(['Present','Absent']);


# In[33]:


round((df.groupby('gender')['no_show'].value_counts())/len(df)*100)


# In[34]:


round((df.groupby('gender')['no_show'].value_counts().unstack())/len(df)*100).plot(kind = 'bar')
plt.legend(['Showed_up','Did not show up'])
plt.title('Percentage of both gender attendance')


# There are obviously more females than male captured in this dataset. To the untrained eye, it seems there isnt much difference between how both gender showed up. They seemes at par, therefore we can't conclude if their gender actually makes a difference.

# ### Research Question 3 : Some patients received sms while others didn't. Would patients that received sms show up more than the others that never received an sms?

# In[35]:


df['sms_received'].value_counts()


# In[36]:


df.sms_received.value_counts(normalize=True)*100


# In[37]:


df_sms = df.groupby('sms_received')['no_show'].value_counts().unstack()
df_sms


# In[38]:


sms_percent= round(df.groupby('sms_received')['no_show'].value_counts()/len(df['sms_received'])*100).unstack()
sms_percent


# In[39]:


df_sms16 = df.groupby('sms_received')['no_show'].value_counts().unstack()
df_sms16.index = ['Received Sms','Did not Receive Sms ']
df_sms16.plot(kind = 'bar', figsize = (8,6), stacked = True)
plt.title("Patients that received sms and showed up/didnt show up")
plt.legend(["showed","missed"])
plt.ylabel("Total")
plt.xlabel("Status")


# The percentage of patients that got an sms is 32% while only 9% of these same set of patients managed to show up. These shows sending sms was not as effective in getting patients to meet up with their scheduled appointments. 

# <a id='conclusions'></a>
# ## Conclusions

# 1. Most of the patients in this dataset actually showed up for their appointments.
# 
# 2. Sms has little to no impact in how the patients showed up for their appointment. the hospitals might likely look for alternatives, maybe calls also.
# 
# 3. Age only shows positive signs between age range of 60-100. 
# 
# 4. Finally, Gender also doesnt have any visible effect in their ability in honouring scheduled appointments. It is virtual the same result from gender even though we had more female patients in our dataset.

# ## Limitations

# 1. I had used the No-show appointment dataset for my analysis and worked with age, gender and sms recived. Our analysis is limited to only the provided dataset. For example,it have been interesting to have the distance from patients home to the hospital's location, this would have given us insights about proximity from the hospital and back home.
# 
# 2.  A variable(Handicap) have not been measured in the way originally thought. It had lots of rows with unreasonable values, so it couldn't be used for study.
# 
# 3. More informations also needs to be provided on major sickness/ diseases that would prompt patients to book appointments and surely show up for thier appointments.

# In[40]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




