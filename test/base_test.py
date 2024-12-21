import unittest
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, utils_dir)
import process_data

'''
Base test class for testing titanic data
'''
class baseTestTitanic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
       process_data.process_data() 
       cls.X_train = process_data.X_train
       cls.y_train = process_data.y_train
       cls.X_test = process_data.X_test
       cls.y_test = process_data.y_test
       cls.features = process_data.features
