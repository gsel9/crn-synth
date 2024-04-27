""" Functionality to integrate synthcity-specific routines.
"""
# NOTE: avoids segmentation error when loading synthcity 
import torch 

from synthcity.plugins import Plugin
from synthcity.plugins.core.dataloader import DataLoader


def dataframe_to_dataloader():
    pass 


#if isinstance(data_loader, DataLoader):
def dataloader_to_dataframe():
    pass 


#if isinstance(generator, Plugin):
def save_synthcity_generator():
    pass 