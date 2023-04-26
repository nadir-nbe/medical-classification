import os
import re
import sys

path = './data/'





file = open('/home/nadir/workspace/machine_learning/ClinicalBERT/data_results/results_train_binary.txt', "r")
file_result = open('/home/nadir/workspace/machine_learning/ClinicalBERT/data_results/results_train_binary_cleaned.txt', "w")





contents = file.readlines()
for sentence in contents:
    if len(sentence)>50:
        print(sentence)
        file_result.write(sentence)
file.close()
file_result.close()


sys.exit(0)