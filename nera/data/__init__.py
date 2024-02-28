"""
    module: data
    author: jakub sukdol

    basic functionalities for data management
"""

# expose some members of protected module parts
from ._data_acquisition import DataAcquisition
from ._macros import FROM_CSV, FROM_NBA_STATS, FROM_FLASHSCORE
from ._data_saving_loading import *
from ._data_transformation import DataTransformation
