# Import Open CV or cv2
import cv2 as c
from random import randrange

#  Define the CasCade Classifier and load the data and algorithms
t_f_d = c.CascadeClassifier("harrascade.xml")

#  Get an image of a person
img = c.imread("RJD.jpg")
webcam = c.VideoCapture(0)

# Iterate to frames forever
while True:
    # Read current frame 
    sfr, fr = webcam.read()

    # Making the images all black and white
    gimg = c.cvtColor(fr, c.COLOR_BGR2GRAY)

    # detect
    fc = t_f_d.detectMultiScale(gimg)

    for (x, y, w, h) in fc:
        c.rectangle(fr, (x, y), (x + w, y+h), (randrange(256), randrange(256), randrange(256)), 4)


    c.imshow("Saad's Face Detector", fr)
    c.waitKey(1)
