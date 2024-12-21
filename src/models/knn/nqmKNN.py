import numpy as np

class knn():
    def __init__(self, k):
        self.k = k
    
    def encode(self, X) -> np.array:
        '''
        Encodes X into one-hot and then treats each entry like a binary digit for hamming distance
        '''
        encoding = ["".join(row) for row in X.astype(str)]
        return np.array(encoding)

    def hamming_dist(self, x_1: str, x_2: str) :
        '''
        Finds the hamming dist between two vectors, which is simply the number of value inequalities.
        Note, each vector needs its values to be combined into a string beforehand. The length of both vectors should always match.
        '''
        assert len(x_1) == len(x_2), "hamming_dist: length of the two input vectors does not match"
        dist = 0
        for i in len(x_1):
            if x_1[i] != x_2[i]: 
                dist += 1
        return dist
