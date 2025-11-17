import unittest
import pandas as pd
from src.utils.data_loader import load_data, preprocess_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        # Create a dummy csv file
        dummy_data = {'col1': [1, 2], 'col2': [3, 4]}
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv('dummy_test.csv', index=False)
        
        # Test loading the dummy data
        df = load_data('dummy_test.csv')
        self.assertEqual(df.shape, (2, 2))

    def test_preprocess_data(self):
        # Create a dummy dataframe
        dummy_data = {'col1': [1, 2], 'col2': [3, 4]}
        dummy_df = pd.DataFrame(dummy_data)
        
        # Test preprocessing the dummy data
        df = preprocess_data(dummy_df)
        self.assertEqual(df.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
