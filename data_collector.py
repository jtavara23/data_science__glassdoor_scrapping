  
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:47:44 2020
@author: Ken
"""

import glassdoor_scraper as gs 
import pandas as pd 

#path = "C:/Users/Ken/Documents/ds_salary_proj/chromedriver"
path = "F:/ds_salary_project/chromedriver"

jobs = gs.get_jobs('data engineer',100, False, path, 15)
df = pd.DataFrame(jobs)  #This line converts the dictionary object into a pandas DataFrame.
df.to_csv('glassdoor_jobs5.csv', index = False)

