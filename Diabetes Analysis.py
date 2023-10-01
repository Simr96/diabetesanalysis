#!/usr/bin/env python
# coding: utf-8

# In[357]:


import numpy as np
import pandas as pd

patient = pd.read_csv('patients.csv')

treatment = pd.read_csv('treatments.csv')

advDia = pd.read_csv('adverse_reactions.csv')


# In[358]:


patient.sample(2)


# In[359]:


treatment.sample(2)


# In[360]:


advDia.sample(2)


# # Quality
# ### patients table
#         Zip code is a float not a string
#         Zip code has four digits sometimes
#         Tim Neudorf height is 27 in instead of 72 in
#         Full state names sometimes, abbreviations other times
#         Dsvid Gustafsson
#         Missing demographic information (address - contact columns) *(can't clean)*
#         Erroneous datatypes (assigned sex, state, zip_code, and birthdate columns)
#         Multiple phone number formats
#         Default John Doe data
#         Multiple records for Jakobsen, Gersten, Taylor
#         kgs instead of lbs for Zaitseva weight
# ### treatments table
#         Missing HbA1c changes
#         The letter 'u' in starting and ending doses for Auralin and Novodra
#         Lowercase given names and surnames
#         Missing records (280 instead of 350)
#         Erroneous datatypes (auralin and novodra columns)
#         Inaccurate HbA1c changes (leading 4s mistaken as 9s)
#         Nulls represented as dashes (-) in auralin and novodra columns
# ### adverse_reactions table
#         Lowercase given names and surnames
# ### Tidiness
#         Contact column in patients table should be split into phone number and email
#         Three variables in two columns in treatments table (treatment, start dose and end dose)
#         Adverse reaction should be part of the treatments table
#         Given name and surname columns in patients table duplicated in treatments and adverse_reactions tables

# In[ ]:





# In[361]:


patient['given_name'] = patient['given_name'].str.lower()


# In[362]:


patient.isnull().sum()


# In[363]:


patient['zip_code'] = patient['zip_code'].dropna()


# In[364]:


patient[patient.isna().any(axis=1)]


# In[365]:


patient.dropna(axis=0, inplace=True)


# In[ ]:





# In[366]:


patient.info()


# In[367]:


patient['zip_code'] = patient['zip_code'].astype(int)


# In[368]:


#patient['phone_number'] = patient['contact'].str.extract(r'(\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4})', expand=False)


# Extract email addresses and store them in a new column 'email_id'
#patient['email_id'] = patient['contact'].str.extract(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4})', expand=False)

patient['phone_number'] = patient.contact.str.extract('((?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', expand=True)

# [a-zA-Z] to signify emails in this dataset all start and end with letters
patient['email'] = patient.contact.str.extract('([a-zA-Z][a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z])', expand=True)

# Note: axis=1 denotes that we are referring to a column, not a row
patient = patient.drop('contact', axis=1)


# In[369]:


patient.sample(3)


# In[370]:


patient['birthdate'] = pd.to_datetime(patient['birthdate']).dt.year


# In[371]:


patient['phone_number'].isnull().sum()


# In[372]:


patient[['home_no', 'Road_add']] = patient['address'].str.extract(r'^(\d+) (.+)$')


# In[373]:


patient.sample(2)


# In[374]:


patient.drop(columns = [ 'address'] , inplace = True)


# In[375]:


patient.sample(2)


# In[376]:


patient.set_index('patient_id')


# In[377]:


patient.set_index('patient_id').head()


# In[378]:


meanofheight =int(patient['height'].mean())


# In[379]:


patient['height'].mean()


# In[380]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[381]:


patient.info()


# In[382]:


sns.distplot(patient['height'])


# In[383]:


patient['city'].value_counts()


# In[384]:


patient['zip_code'] = patient['zip_code'].astype(str).str.pad(5, fillchar='0')


# In[385]:


patient.sample(3)


# In[386]:


patient.birthdate = pd.to_datetime(patient.birthdate).dt.year   


# In[387]:


patient.sample(2)


# In[388]:


patient.assigned_sex = patient.assigned_sex.astype('category')


# In[389]:


patient.sample(2)


# In[390]:


patient['name'] = (patient['given_name'] + ' ' + patient['surname']).str.lower()
patient.drop(columns = ['given_name' , 'surname'] , inplace = True)


# In[391]:


patient.sample(2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[392]:


treatment = pd.melt(treatment, id_vars=['given_name', 'surname', 'hba1c_start', 'hba1c_end', 'hba1c_change'], 
                           var_name='treatment', value_name='dose')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[393]:


treatment = treatment[treatment.dose != "-"]

treatment['dose_start'], treatment['dose_end'] = treatment['dose'].str.split(' - ', 1).str
treatment.drop('dose', axis=1)


# In[394]:


treatment['dose_start'] = treatment['dose_start'].str.extract(r'(\d+)')


# In[395]:


treatment.sample(2)


# In[396]:


treatment['dose_end'] = treatment['dose_end'].str.extract(r'(\d+)')


# In[397]:


treatment.drop(columns  = 'dose', inplace = True)


# In[398]:


treatment.sample(1)


# In[399]:


treatment['dose_end'] = treatment['dose_end'].str.extract(r'(\d+)')


# In[400]:


treatment.sample(11)


# In[401]:


treatment


# In[402]:


treatment['dose_end'].fillna(0, inplace=True)


# In[403]:


treatment


# In[404]:


treatment['hba1c_change'] = treatment['hba1c_start'] - treatment['hba1c_end']


# In[405]:


treatment.sample(2)


# In[421]:


treatment['name'] = (treatment['given_name'] + ' ' + treatment['surname']).str.lower()
treatment.drop(columns = ['given_name' , 'surname'] , inplace = True)


# In[422]:


treatment.sample(2)


# In[ ]:





# In[406]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[407]:


sns.histplot(treatment['hba1c_change'], bins=9 , kde=True)


# In[408]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[409]:


min_max_scaler = MinMaxScaler()

treatment['hba1c_normalized'] = min_max_scaler.fit_transform(treatment[['hba1c_change']])



# In[410]:


sns.histplot(treatment['hba1c_normalized'], bins=10, kde=True)


# In[425]:


treatment.head(2)


# In[412]:


merged_df = treatment.merge(advDia[['umm']], left_index=True, right_index=True, how='left')


# In[413]:


advDia['name'] = advDia['given_name'] + ' ' + advDia['surname']


# In[414]:


advDia.drop(columns = ['given_name', 'surname'] , inplace = True)


# In[424]:


advDia.head(2)


# In[426]:


merged_df = treatment.merge(advDia[['adverse_reaction']], left_index=True, right_index=True, how='left')


# In[419]:


advDia.isnull().sum()


# In[420]:


treatment


# In[428]:


merged_df.head(20)


# In[ ]:





# In[ ]:





# In[ ]:




