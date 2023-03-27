
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
