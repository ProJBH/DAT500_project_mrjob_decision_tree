#!/usr/bin/env python3
from mrjob.job import MRJob
import random
from datetime import datetime
import csv

class sklearn_train_preparation_map(MRJob):   
    def mapper(self, _, line):
        random.seed(datetime.now())
        key = random.random()
        rows = csv.reader([line])
        for row in rows:
            yield (key+' test',row)
        
    def reducer(self, key, values):
        yield (key, list(values))
    
if __name__ == '__main__':
    sklearn_train_preparation_map.run()