# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.willow_garage
import numpy as np

# Set up willow_garage_2011
willow_garage_devkit_path = '/home/ubuntu/WillowDataset'
for split in ['train', 'validation', 'test']:
    name = '{}_{}'.format('willow_garage', split) 
    print datasets.willow_garage
    #imdb = datasets.willow_garage.willow_garage('train', '2011', willow_garage_devkit_path)
    #print imdb
    #__sets[name] = imdb
    __sets[name] = (lambda split=split: datasets.willow_garage.willow_garage(split, '2011', willow_garage_devkit_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
