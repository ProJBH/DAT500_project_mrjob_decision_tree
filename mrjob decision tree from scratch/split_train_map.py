from mrjob.job import MRJob
import random
from datetime import datetime

class split_map(MRJob):
    def mapper(self, _, line):
        random.seed(datetime.now())
        tag = random.random()
        line = line.strip()
        row = line.split(',')
        if row[0] != "customer_id":
            num = random.randint(0,10)
            if num <=1:
                key = 'Test'+str(tag)
                yield (key,row)
            else:
                key = 'Train'+str(tag)
                yield (key,row)
        
    def reducer(self, key, values):
        yield key, list(values)   
    
if __name__ == '__main__':
    split_map.run()
        