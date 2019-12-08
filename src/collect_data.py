import os
import sys
from glob import glob
from multiprocessing import Pool, cpu_count
from itertools import repeat

import cv2
import numpy as np
from face_recognition import face_locations

# Draw face location
def draw_face_locations(img, face_locs):
    len_locs = len(face_locs)
    img_new = np.array(img)
    if len_locs:
        cl_blue = (255, 0, 0)
        i = 0
        for (x,y,w,h) in face_locs:
            print ("x = ",x)
            print ("y = ",y)
            print ("w = ",w)
            print ("h = ",h)
            c1 = (h, x)
            c2 = (y, w)
            cv2.rectangle(img_new, c1, c2, cl_blue, 3)
            i += 1
    return img_new

# Detect blured images
def detect_blur(img, thresh=None):
    n_channels = img.shape[2]
    var = 0
    for c in range(n_channels):
        value = img[..., c]
        var += cv2.Laplacian(value, cv2.CV_64F).var()    # var() compute variance of Laplacian 
    var = var/n_channels
    if thresh==None:
        return var
    else:
        return True if var < thresh else False, var
    
    
if __name__ == "__main__":
    name_id = input(">> Enter your ID: ")
    # Invalid syntax
    if len(name_id) != 7:
        sys.exit("Invalid name ID!!!")
        
    # Check the existance
    folder = os.path.join("../pre_dataset/raw_dataset", name_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
        save_id = 0
    else:
        key = input(">> Name ID has already existed. Do you want to append? [y/n] ")
        save_id = len(glob(os.path.join(folder, "*.*")))
        if key=="n" or key=="N":
            sys.exit("Exit the application.")
            
    #.................................................        
    print("Wait for calibrating the blurness...")
    cap = cv2.VideoCapture(0)
    frames = []
    for i in range(50):
        _, frame = cap.read()
        frames.append(frame)
    
    pools = Pool(processes=cpu_count())
    args = zip(frames, repeat(None))
    blurness_vars = pools.map(detect_blur, frames)
    
    blurness_thres = sum(blurness_vars) / len(blurness_vars) * 0.90
    print("Finish calibrating the blurness: %f", blurness_thres)
    print("--------------------------------------------------------\n")
    
    frame_idx = 0
    count = 0
    while(cap.isOpened()):
        # Read frame
    
        _, frame = cap.read()
        print("[FrameID %d] " % frame_idx)
        frame_idx += 1
    
        # Detect blur
        blurness, var = detect_blur(frame, blurness_thres)
    
        # Detect and draw face
        face_locs = face_locations(frame)
        img = draw_face_locations(frame, face_locs)
    
        # Show the frame
        cv2.imshow('Please change face direction', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count > 50:
            break
    
         # Save the frame
        if (blurness == False) and (len(face_locs)==1):
            filename = os.path.join(folder, "%s_%d.png" % (name_id, save_id))
            cv2.imwrite(filename, frame)
            save_id += 1
            count += 1
            print("Number of saved frames: %d" % (save_id))
            print("--------------------------------------------------------\n")
    cap.release()
    cv2.destroyAllWindows()
    