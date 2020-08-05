from __future__ import absolute_import

from .invariance import InvNet

from .triplet import TripletLoss ,TripletLoss2,SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy


__all__ = [
    'InvNet',
    'TripletLoss',
    'TripletLoss2',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy'
]
