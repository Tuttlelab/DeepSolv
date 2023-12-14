#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:59:10 2023

@author: rkb19187
"""

# -*- coding: utf-8 -*-

class Logger:
    def __init__(self, logfile=None, verbose=True):
        if type(logfile) == str:
            self.Logfile = open(logfile, 'w')
        else:
            self.Logfile = False
        
        self.verbose = bool(verbose)
    
    def Log(self, string):
        string = str(string)
        if self.Logfile != False:
            self.Logfile.write(string)
            self.Logfile.write("\n")
            self.Logfile.flush()
        if self.verbose:
            print(string)
        
    def close(self):
        if self.Logfile != False:
            self.Logfile.close()
            self.Logfile = False