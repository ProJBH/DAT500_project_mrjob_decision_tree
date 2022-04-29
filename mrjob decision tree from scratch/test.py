from mrjob.job import MRJob
class t(MRJob):
    def mapper(self, _, value):
        columns = value.split(',')
        yield len(columns),1
    
    def reducer(self, key, values):
        yield key, sum(values)