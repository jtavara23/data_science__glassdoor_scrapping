# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:27:06 2020
@author: Ken
"""

import pandas as pd 

df = pd.read_csv('glassdoor_jobs_total.csv')
"""
#salary parsing 
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2
"""

#Company name text only
df['company'] = df.apply(lambda x: x['Company Name'][:-4] if x['Company Name'][-1].isdigit() else x['Company Name'], axis = 1)


#Location 
#print(df.Location.value_counts())


#Age of company 
df['age'] = df.Founded.apply(lambda x: x if x <1 else 2020 - x)
#print(df.age)

#""" parsing of job description (python, etc.)""""
#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
print(df.python_yn.value_counts())

#r studio 
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
print(df.R_yn.value_counts())

#spark 
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
print(df.spark.value_counts())

#aws 
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
print(df.aws.value_counts())

#spark 
df['gcp'] = df['Job Description'].apply(lambda x: 1 if 'gcp' in x.lower() else 0)
print(df.gcp.value_counts())

df['azure'] = df['Job Description'].apply(lambda x: 1 if 'azure' in x.lower() else 0)
print(df.azure.value_counts())

df['sql'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
print(df.sql.value_counts())

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
print(df.excel.value_counts())

#print(df.columns)

df_out = df.drop(['Salary Estimate','Headquarters','Competitors', 'Company Name'], axis =1)

df_out.to_csv('data_cleaned.csv',index = False)
