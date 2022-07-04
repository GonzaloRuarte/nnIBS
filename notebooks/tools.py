import os
import numpy as np
import pandas as pd

#class Interiors():
#    def __init__(self) -> None:

def get_dataframe_from_(csv_file):
    df = pd.read_csv(csv_file)
    return df        