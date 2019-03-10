#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Sun Mar 10 18:07:57 2019
"""
from boxx import *


r = 3

def getCycle(r):
    y,x = np.mgrid[-r:r+1,-r:r+1]
    distance = (y**2 + x**2)**.5
    cyc = distance<distance[r,0]
    return cyc
    show-[distance, cyc]
if __name__ == "__main__":
    pass
    
    
    
