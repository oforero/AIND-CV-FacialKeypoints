#!/usr/bin/env python3

"""
video_face_detection.py
Python3 and Qt5 
"""
 
### Add face and eye detection to this laptop camera function 
# Make sure to draw out all faces/eyes found in each frame on the shown video feed

import cv2
import sys
import time 

from face_eye_detector import detect_faces_and_eyes, blur_faces_in_image
from face_eye_detector import load_model, detect_faces_and_markers, detect_faces_add_glasses

POST_FNS = {
    'BLUR': blur_faces_in_image,
    'EYES': detect_faces_and_eyes,
    'MARKERS': detect_faces_and_markers,
    'GLASSES': detect_faces_add_glasses
}

# wrapper function for face/eye detection with your laptop camera
def laptop_camera_go(outfile, post='BLUR', model_file='my_model.h5'):

    face_tr_Fn = POST_FNS[post]
    if post in ['MARKERS', 'GLASSES']:
        model = load_model(model_file)
        face_tr_Fn = lambda img: POST_FNS[post](model, img)

    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)
    print("VC: ", vc)

    # Try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False
    
    # Create instance of video writer if a file parameter is passed
    out = None
    if outfile:
        print("Writing video: ", outfile, frame.shape)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15.0
        out = cv2.VideoWriter(outfile, fourcc, fps, (frame.shape[1], frame.shape[0])) 
    
    # Keep the video stream open
    while rval:

        # Plot the image from camera with all the face and eye detections marked
        frame_with_detections = face_tr_Fn(frame)
        cv2.imshow("face detection activated", frame_with_detections)
        if outfile:
            out.write(frame_with_detections)

        # Exit functionality - press any key to exit laptop video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vc.release()
            if outfile:
                print("Saving video")
                out.release()
            # Destroy windows 
            cv2.destroyAllWindows()
            
            # Make sure window closes on OSx
            for i in range (1,5):
                cv2.waitKey(1)
            return
        
        # Read next frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()    


def main(argv):
    if len(argv) == 1:
        argv.append(False)
        argv.append('MARKERS')
    
    if len(argv) == 2:
        argv.append('MARKERS')
    
    _, out_file, effect = argv

    laptop_camera_go(out_file, effect)

if __name__ == '__main__':
    main(sys.argv)
