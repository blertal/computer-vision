# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:30:09 2017

@author: blerta
"""

from test import test

charts = True

testFilename   = 'test1.bmp'
testGTFilename = 'test1_gt.pkl'
test( testFilename, testGTFilename, charts)

testFilename   = 'test2.bmp'
testGTFilename = 'test2_gt.pkl'
test( testFilename, testGTFilename, charts)
