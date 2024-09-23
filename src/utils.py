from typing import Dict, List
import pandas as pd
import time

# ML
from torch.utils.data.dataset import Dataset

# ==================================== Pandas ====================================
class PandasToTorchDataset(Dataset):
    """
    Converts a pandas DataFrame to a torch dataset object
    """

    def __init__(self, data: pd.DataFrame, columns: List[str]):
        """
        Args :
            - data      : DataFrame object to use in the torch dataset
            - columns   : Columns in the DataFrame to consider when indexing
        """
        self.data = data
        self.columns = columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[self.columns].iloc[idx].tolist()


def pd_summary(df: pd.DataFrame):
    """
    Prints a summary of a pandas DataFrame (number of rows and columns)

    Args :
        - df    : DataFrame to show the summary of
    """

    print('Number of rows :\t\t', len(df))
    print('Columns :\t\t', list(df.columns))

# ==================================== General ====================================

def time_function(function):
    """
    Computes the running time of a function

    Args :
        - function  : Function to time

    Returns
    Tuple containing the time of the function as well as the result
    """
    start = time.time()
    result = function()
    end = time.time()
    return end - start, result

def normalize_frequency_dict(frequency_dict: Dict[str, int]):
    """
    Takes a dictionary where the keys are a string and the values are the occurrence
    of that string (in a text for example) and normalizes the occurrence using the
    total occurrences
    """
    frequencies = {}

    total = 0
    for key, count in frequency_dict.items():
        total += count
        frequencies[key] = count

    for key, value in frequencies.items():
        frequencies[key] = value / total

    return frequencies
