'''Fundamental:The return value of the eye aspect ratio will be  
approximatelyconstant when the eye is open. 
The value will then rapid decrease towards zero during a blink.
monitoring the eye aspect ratio to see 
if the value falls but does not increase again, 
thus implying that the person has closed their eyes.'''
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav


#import necessary pachage
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


#define sound_alarm function
def sound_alarm(path):
    playsound.playsound(path)
    
#define the eye_aspect_ratio function
def eye_aspect_ratio(eye):
    #compute the euclidean distances between the two sets of
    #vertical eye landmarks (x,y)-coordinates
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    # compute the euclidean distances between the two sets of
    # horizontal eye landmarks(x,y)-coordinates
    C = dist.euclidean(eye[0],eye[3])
    # compute the eye aspect ratio 
    ear = (A + B)/(2.0 * C)
    # return the eye  aspect ratio
    return ear

# parse our command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmarks predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
    help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int,default=0,
    help="index of webcam on system")
# --webcam : This integer controls the 
# index of your built-in webcam/USB camera.
args = vars(ap.parse_args())#执行命令参数


''' define a few important variables'''

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
eye_ar_thresh = 0.3 #adjusting the two constants to
eye_ar_constant_frames = 15 #change the  Sensitivity of the drowsiness detector


# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
alarm_on = False


'''The dlib library ships with a Histogram of Oriented Gradients-based face detector 
along with a facial landmark predictor — 
we instantiate both of these in the following code block'''

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

print ("[INF0] loading facial landmarks predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively from a set of facial landmarks
(LStart, LEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(RStart, REnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# Using these indexes, we’ll easily be able to 
# extract the eye regions via an array slice.

'''core of our drowsiness detector'''

# start the video stream thread
print("[INF0] start video stream thread..." )
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    '''apply facial landmark detection to localize 
    each of the important regions of the face'''
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[LStart:LEnd]
        rightEye = shape[RStart:REnd]
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEar + rightEar)/2


    # visualize each of the eye regions 

    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull],-1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

    # Check the driver if to show symptoms of drowsiness


    # check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
        if ear < eye_ar_thresh:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >=eye_ar_constant_frames:
                # if the alarm is not on, turn it on
                if not alarm_on:
                    alarm_on = True


                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background

                    if  args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                            args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!!!", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm

        else:
            COUNTER=0
            alarm_on=False

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    #show the frames
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
















