import os
import re
import sys
import pandas as pd
path = './data/'

files = []

classes =  {' Allergy / Immunology':'0',
 ' Autopsy':'1',
 ' Bariatrics':'2',
 ' Cardiovascular / Pulmonary':'3',
 ' Chiropractic':'4',
 ' Consult - History and Phy.':'5',
 ' Cosmetic / Plastic Surgery':'6',
 ' Dentistry':'7',
 ' Dermatology':'8',
 ' Diets and Nutritions':'9',
 ' Discharge Summary':'10',
 ' ENT - Otolaryngology':'11',
 ' Emergency Room Reports':'12',
 ' Endocrinology':'13',
 ' Gastroenterology':'14',
 ' General Medicine':'15',
 ' Hematology - Oncology':'16',
 ' Hospice - Palliative Care':'17',
 ' Lab Medicine - Pathology':'18',
 ' Letters':'19',
 ' Nephrology':'20',
 ' Neurology':'21',
 ' Neurosurgery':'22',
 ' Obstetrics / Gynecology':'23',
 ' Office Notes':'24',
 ' Ophthalmology':'25',
 ' Orthopedic':'26',
 ' Pain Management':'27',
 ' Pediatrics - Neonatal':'28',
 ' Physical Medicine - Rehab':'29',
 ' Podiatry':'30',
 ' Psychiatry / Psychology':'31',
 ' Radiology':'32',
 ' Rheumatology':'33',
 ' SOAP / Chart / Progress Notes':'34',
 ' Sleep Medicine':'35',
 ' Speech - Language':'36',
 ' Surgery':'37',
 ' Urology':'38'}

selected_classes =  {' Gastroenterology':'0',' Neurology':'1',' Orthopedic':'2',' Radiology':'3',' Urology':'4',' Obstetrics / Gynecology':'5',' Discharge Summary':'6',' ENT - Otolaryngology':'7',' Hematology - Oncology':'8',' Neurosurgery':'9'}
print(selected_classes.keys())
df_train = pd.read_csv('./data_results/mtsamples.csv', delimiter = ',')
df_train_data = df_train['sentence'].values
df_train_target = df_train['Class'].values
print(df_train_target)
i=0
file_result = open('data_results/mtsamples_cleaned_fix.csv', "w")
for sentence in df_train_data:
    if (len(str(sentence))> 50) and df_train_target[i] in selected_classes.keys():
        file_result.write(sentence +'\t' +str(selected_classes[df_train_target[i]] + "\n"))
    i=i+1
sys.exit(0)