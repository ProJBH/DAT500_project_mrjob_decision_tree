#!/usr/bin/env python3
import re
from mrjob.job import MRJob
import pandas as pd
import random
from datetime import datetime
import csv

class preprocessing(MRJob):
    
    def mapper(self, _, line):
        random.seed(datetime.now())
        key = random.random()
        #row = line.split(',')
        rows = csv.reader([line])
        for row in rows:
            if row[1] != "gender":
                customer_id = row[0]
                gender = row[1]
                status_x = row[2]
                verified_x = row[3]
                create_at_x = row[4].split(' ')[0]
                location_number = row[6]
                updated_at_x = row[5].split(' ')[0]
                location_number = row[6]
                location_type = row[7]
                vendor_id = row[10]
                latitude_x = row[8]
                longitude_x = row[9]
                latitude_y = row[12]
                longitude_y = row[13]
                vendor_category_en = row[14]
                delivery_charge = row[15]
                serving_distance = row[17]
                is_open = row[18]
                prepration_time = row[21]
                commission = row[22]
                is_akeed_delivering = row[23]
                discount_percentage = row[24]
                status_y = row[25]
                verified_y = row[26]
                rank = row[27]
                language = row[28]
                vendor_rating = row[29]
                sunday_from_time1 = pd.to_datetime(row[30])
                sunday_to_time1 = pd.to_datetime(row[31])
                sunday_from_time2 = pd.to_datetime(row[32])
                sunday_to_time2 = pd.to_datetime(row[33])
                temSun = ((sunday_to_time1-sunday_from_time1)+(sunday_to_time2-sunday_from_time2))
                temSun = temSun.total_seconds()/3600
                sunday = str(round(temSun,1)).split(' ')[-1]
                monday_from_time1 = pd.to_datetime(row[34])
                monday_to_time1 = pd.to_datetime(row[35])
                monday_from_time2 = pd.to_datetime(row[36])
                monday_to_time2 = pd.to_datetime(row[37])
                temMon = ((monday_to_time1-monday_from_time1)+(monday_to_time2-monday_from_time2))
                temMon = temMon.total_seconds()/3600
                monday = str(round(temMon,1)).split(' ')[-1]
                tuesday_from_time1 = pd.to_datetime(row[38])
                tuesday_to_time1 = pd.to_datetime(row[39])
                tuesday_from_time2 = pd.to_datetime(row[40])
                tuesday_to_time2 = pd.to_datetime(row[41])
                temTue = ((tuesday_to_time1-tuesday_from_time1)+(tuesday_to_time2-tuesday_from_time2))
                temTue = temTue.total_seconds()/3600
                tuesday = str(round(temTue,1)).split(' ')[-1]
                wednesday_from_time1 = pd.to_datetime(row[42])
                wednesday_to_time1 = pd.to_datetime(row[43])
                wednesday_from_time2 = pd.to_datetime(row[44])
                wednesday_to_time2 = pd.to_datetime(row[45])
                temWed = ((wednesday_to_time1-wednesday_from_time1)+(wednesday_to_time2-wednesday_from_time2))
                temWed = temWed.total_seconds()/3600
                wednesday = str(round(temWed,1)).split(' ')[-1]
                thursday_from_time1 = pd.to_datetime(row[46])
                thursday_to_time1 = pd.to_datetime(row[47])
                thursday_from_time2 = pd.to_datetime(row[48])
                thursday_to_time2 = pd.to_datetime(row[49])
                temThur = ((thursday_to_time1-thursday_from_time1)+(thursday_to_time2-thursday_from_time2))
                temThur = temThur.total_seconds()/3600
                thursday = str(round(temThur,1)).split(' ')[-1]
                friday_from_time1 = pd.to_datetime(row[50])
                friday_to_time1 = pd.to_datetime(row[51])
                friday_from_time2 = pd.to_datetime(row[52])
                friday_to_time2 = pd.to_datetime(row[53])
                temFri = ((friday_to_time1-friday_from_time1)+(friday_to_time2-friday_from_time2))
                temFri = temFri.total_seconds()/3600
                friday = str(round(temFri,1)).split(' ')[-1]
                saturday_from_time1 = pd.to_datetime(row[54])
                saturday_to_time1 = pd.to_datetime(row[55])
                saturday_from_time2 = pd.to_datetime(row[56])
                saturday_to_time2 = pd.to_datetime(row[57])
                temSat = ((saturday_to_time1-saturday_from_time1)+(saturday_to_time2-saturday_from_time2))
                temSat = temSat.total_seconds()/3600
                saturday = str(round(temSat,1)).split(' ')[-1]
                primary_tags = row[58]
                primary_tags =re.sub(r"[\\\'\"]", '', primary_tags)
                open_close_flags = row[59]
                vendor_tag = row[60]
                tem = vendor_tag.split('-')
                vendor_tag_sum = str(len(tem))
                created_at_y = row[65].split(' ')[0]
                updated_at_y = row[66].split(' ')[0]
                device_type = row[67]
                location_number_obj = row[69]
                target = row[72]
                tu = (customer_id,gender,status_x,verified_x,create_at_x, updated_at_x,location_number,location_type,vendor_id,latitude_x,longitude_x,latitude_y,longitude_y,vendor_category_en,delivery_charge,serving_distance,is_open,prepration_time,commission,is_akeed_delivering,discount_percentage,status_y,verified_y,rank,language,vendor_rating,sunday,monday,tuesday,wednesday,thursday,friday,saturday,primary_tags,open_close_flags,vendor_tag_sum,created_at_y,updated_at_y,device_type,location_number_obj,target)
            #lst = ["customer_id","gender","status_x","verified_x","create_at_x","updated_at_x","location_number","location_type","vendor_id","latitude_x","longitude_x","latitude_y","longitude_y","vendor_category_en","delivery_charge","serving_distance","is_open","prepration_time","commission","is_akeed_delivering","discount_percentage","status_y","verified_y","rank","language","vendor_rating","sunday","monday","tuesday","wednesday","thursday","friday","saturday","primary_tags","open_close_flags","vendor_tag_sum","created_at_y","updated_at_y","device_type","location_number_obj","target"]
                yield (key,tu)
        
    def reducer(self, key, values):
        yield (key, list(values))
    
if __name__ == '__main__':
    preprocessing.run()