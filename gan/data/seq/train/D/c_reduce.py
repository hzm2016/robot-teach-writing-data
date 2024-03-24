import cv2
import glob

for filename in glob.glob('./*.jpg'):
    infile = cv2.imread(filename)
    infile = cv2.cvtColor(infile, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename, infile)
