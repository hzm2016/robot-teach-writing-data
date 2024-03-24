import cv2
import glob
import os

for filename in glob.glob('*.jpg'):
    index = int(filename[:-4])
    if index < 1000:
        os.system('cp ' + filename + ' ../E/')
    #infile = cv2.imread(filename)
    #infile = cv2.cvtColor(infile, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(filename, infile)
