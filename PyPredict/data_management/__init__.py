"""
    module: data_management
    author: jakub sukdol

    basic functionalities for data management
"""

# expose some members of protected module parts
from .data_acquisition import DataAcquisition
from ._macros import FROM_CSV, FROM_NBA_STATS, FROM_FLASHSCORE
from ._data_saving_loading import *
from .data_transformation import DataTransformation
