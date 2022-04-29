#!/usr/bin/env python3
import sys
import pandas as pd
import zipimport
import numpy as np
import ast

importer = zipimport.zipimporter('DT.mod')
dt = importer.load_module('DecisionTree')
column_name = ['customer_id','gender','status_x','verified_x','create_at_x','updated_at_x','location_number','location_type','vendor_id','latitude_x','longitude_x','latitude_y','longitude_y','vendor_category_en','delivery_charge','serving_distance','is_open','prepration_time','commission','is_akeed_delivering','discount_percentage','status_y','verified_y','rank','language','vendor_rating','sunday','monday','tuesday','wednesday','thursday','friday','saturday','primary_tags','open_close_flags','vendor_tag_sum','created_at_y','updated_at_y','device_type','location_number_obj','target']
trees = []
csv_context = ()
for line in sys.stdin:
    if line.find('tree') == 0:
        line = line.strip()
        line = line.split('\t')
        tree = line[1]
        tree = ast.literal_eval(tree)
        trees.append(tree)
    else:
        line = line.strip()
        row = line.split(',')
        if 'customer_id' not in str(row):
            added_value_tuple = (row,)
            csv_context = csv_context + added_value_tuple

df = pd.DataFrame(csv_context, columns=column_name)
df = df.astype({'sunday': np.float64, 'monday': np.float64, 'tuesday': np.float64, 'wednesday': np.float64, 'thursday': np.float64, 'friday': np.float64, 'saturday': np.float64, 'vendor_tag_sum': np.float64, 'longitude_x': np.float64, 'longitude_y': np.float64, 'latitude_x': np.float64, 'latitude_y': np.float64, 'vendor_rating': np.float64})
for i in range (len(trees)):
    accuracy = dt.calculate_accuracy(df,trees[i])
    print('%s\t%s'%('accuracy',accuracy))
