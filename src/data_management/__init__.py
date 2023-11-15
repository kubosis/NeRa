"""
    module: data_management
    author: jakub sukdol

    basic functionalities for data management
"""

# expose some members of protected module parts
from src.data_management._Data_acquisition import DataAcquisition
from src.data_management._macros import FROM_CSV, FROM_NBA_STATS, FROM_FLASHSCORE
from src.data_management._data_saving_loading import *
