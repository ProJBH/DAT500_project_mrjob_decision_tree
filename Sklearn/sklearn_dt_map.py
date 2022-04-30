#!/usr/bin/env python3
import sys
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import subprocess

training_set = ()
testing_set = ()
column_name = ['customer_id','gender','status_x','verified_x','created_at_x','updated_at_x','location_number','location_type','latitude_x','longitude_x','id','authentication_id','latitude_y','longitude_y','vendor_category_en','vendor_category_id','delivery_charge','serving_distance','is_open','OpeningTime','OpeningTime2','prepration_time','commission','is_akeed_delivering','discount_percentage','status_y','verified_y','rank','language','vendor_rating','sunday_from_time1','sunday_to_time1','sunday_from_time2','sunday_to_time2','monday_from_time1','monday_to_time1','monday_from_time2','monday_to_time2','tuesday_from_time1','tuesday_to_time1','tuesday_from_time2','tuesday_to_time2','wednesday_from_time1','wednesday_to_time1','wednesday_from_time2','wednesday_to_time2','thursday_from_time1','thursday_to_time1','thursday_from_time2','thursday_to_time2','friday_from_time1','friday_to_time1','friday_from_time2','friday_to_time2', 'saturday_from_time1','saturday_to_time1','saturday_from_time2','saturday_to_time2','primary_tags', 'open_close_flags','vendor_tag','vendor_tag_name','one_click_vendor','country_id','city_id','created_at_y','updated_at_y','device_type','display_orders','location_number_obj','id_obj','CID X LOC_NUM X VENDOR','target']

for line in sys.stdin:
    line = line.strip()
    values = line.split('\t')
    key = values[0].split(' ')
    if key == 'train':
        value = values[1].split(',')
        training_set = training_set + (row,)
    elif key == 'test':
        value = values[1].split(',')
        testing_set = testing_set + (row,)

df_train = pd.DataFrame(training_set, columns=column_name)
df_test = pd.DataFrame(testing_set, columns=column_name)
X = df_train.drop("target", axis=1)
y = df_train.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)        
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

for i in range(len(df_test))
    test_pred = clf.predict(df_test[i])
    df_test[i].append(test_pred)
    print(df_test[i])