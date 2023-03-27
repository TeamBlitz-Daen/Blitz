
import os
import pandas as pd 
import matplotlib.pyplot as plot 
import numpy as np

#reading CSV file into DataFrame 

dataset1 = pd.read_csv('Dataset1.csv') 

pd.set_option('display.max_columns', None)

dataset1.head(5)


# splitting the column 'pastPregnancy' by > and storing it in a new dataframe

temp = dataset1['pastPregnancy'].str.split('>', expand=True)

# renaming required columns from index values

temp = temp.rename(columns={temp.columns[1]: 'A'})
temp = temp.rename(columns={temp.columns[2]: 'B'})
temp = temp.rename(columns={temp.columns[3]: 'C'})

# separating first character in the columns

temp['para'] = temp['A'].str[:1]
temp['living'] = temp['B'].str[:1]
temp['gravida'] = temp['C'].str[:1]

# cleaning the data

temp['para'] = temp['para'].str.replace('l','0')
temp['living'] = temp['living'].str.replace('g','0')
temp['gravida'] = temp['gravida'].str.replace('e','0')

# converting object type columns to integer type

temp['para'] = temp['para'].astype(str).astype(int)
temp['gravida'] = temp['gravida'].astype(str).astype(int)
temp['living'] = temp['living'].astype(str).astype(int)

# calculating gravida-para

dataset1['previous_unsuccessful_pregnancies'] = temp['gravida'] - temp['para']



#merging the DATASETS

df2 = pd.read_csv("Dataset2.csv")
df3 = pd.read_csv("Dataset3.csv",header=None)
col_names =["patient_id","appointment_type","appointment_status","clinical_observations_1","diagnosis_title","appointment_reason_1","start_time","end_time","info"]
df3.columns = col_names
df2= df2.drop(["appointment_type","appointment_status","diagnosis_title","start_time","end_time"],axis=1)
df3= df3.drop(["appointment_type","appointment_status","diagnosis_title","start_time","end_time"],axis=1)
def concat_values(series):
    return ','.join(str(x) for x in series)

df2 = df2.groupby('patient_id')[['doctor_advice_notes', 'clinical_observations','appointment_reason']].agg(concat_values).reset_index()
df3 = df3.groupby('patient_id')[[ 'clinical_observations_1','appointment_reason_1','info']].agg(concat_values).reset_index()
df_merged = pd.merge(df2, df3, on='patient_id')

df_merged["full_clinical_obs"] =  df_merged['clinical_observations']+df_merged['clinical_observations_1']
df_merged["full_Appointment_reasons"] =  df_merged['appointment_reason']+df_merged['appointment_reason_1']
df_merged = df_merged.drop(["clinical_observations","clinical_observations_1","appointment_reason","appointment_reason_1"],axis=1)
df =  pd.merge(dataset1,df_merged, on="patient_id")
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], format='%d/%m/%Y')
df['edd'] = pd.to_datetime(df['edd'], format='%d/%m/%Y')
years, months = divmod((df['edd'] - df['date_of_birth']).dt.days, 365)
df['Age_at_pregnancy'] = years.round()

##df.to_csv(r'final_dataset.csv')



#updated from here 
##the main data set is in the variable df, please look the column info 


# updated the created a new dataset with all past medical records as flags "past_medical_df"
df['past_medical_history'] = df['past_medical_history'].apply(str)
all_medical =''
for index, row in df.iterrows():
    if row['past_medical_history']!= "nan":
        all_medical += row['past_medical_history']
    all_medical += ','
list1 = all_medical.split(",")
while '' in list1:
    list1.remove('')

    
list1 = list(set(list1))
past_medical_df = pd.DataFrame(columns = (['patient_id']+list1))

for index, row in df.iterrows():
    i =  row['past_medical_history']
    flag_list = []
    check_list = i.split(',')
    flag_list.append(row["patient_id"])
    for j in list1:
        if i =='nan':
            flag_list.append(0)
        elif j in check_list:
            flag_list.append(1)
        else:
            flag_list.append(0)
    past_medical_df = past_medical_df.append(pd.Series(flag_list, index = (['patient_id']+list1)), 
           ignore_index=True)        