import os
import pandas as pd
import glob
class allfilelist(object):
    def __init__(self):
        self.file = ""
    def allfilesinlist(self, pathroot,type):
        filelist =[]
        findrepeat=[]
        allsub = []

        file_ = (glob.glob(pathroot + "*." + type))

        for f in file_:
           findrepeat.append(f[f.find('\\')+1:])
           filelist.append(f)


        for path, subdirs, files in os.walk(pathroot):

            for sub in subdirs:
                allsub.append(sub)


        for sub in allsub:
            subpath = pathroot + sub
            for path, subdirs, files in os.walk(subpath):
                for file in files:
                    if file not in findrepeat:
                        filename = path + '\\'+ file

                        filelist.append(filename)

        return filelist

