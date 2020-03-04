import cv2
import numpy as np
import os
import sys
from glob import glob
from face_recognition import face_locations

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
    return var < thresh

if __name__ == "__main__":
    name_id = input(">> Enter your name: ")
        
    # Check if name exist
    folder = os.path.join("../dataset/raw_dataset", name_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
        save_id = 0
    else:
        key = input(">> This name has already existed. Do you want to append images? [y/n] ")
        save_id = len(glob(os.path.join(folder, "*.*")))
        if key=="n" or key=="N":
            sys.exit("Exit the application.")
            
    #.................................................        
    print("Get sample to calculate the blur_thres value...")
    cap = cv2.VideoCapture(0)
    # 
    frames = []
    for i in range(50):
        _, frame = cap.read()
        frames.append(frame)

    blur_thres = np.sum([detect_blur(frame) for frame in frames])/len(frames)*0.9
    
    print("blur_thres = %f", blur_thres)
    print("--------------------------------------------------------\n")
    
    frame_no = 0
    count = 0
    while(cap.isOpened()):
        # Read frame
        _, frame = cap.read()
        print("[Frame .No %d] " % frame_no)
        frame_no += 1
    
        # Detect blur
        is_blur = detect_blur(frame, blur_thres)
    
        # Detect and draw face
        face_locates = face_locations(frame)

        # boxed face
        face_exist = len(face_locates)
        img = np.array(frame)
        if face_exist:
            color_blue = (255, 0, 0)
            for (top, right, bottom, left) in face_locates:
                start_point = (left, top)
                end_point = (right, bottom)
                cv2.rectangle(img, start_point, end_point, color_blue, 3)

        # Show the image
        cv2.imshow('Please navigate your face', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count > 10:
            break
    
         # Save the data
        is_multi_face = False
        if (is_blur == False) and (len(face_locates)==1 or is_multi_face):
            filename = os.path.join(folder, "%s_%d.png" % (name_id, save_id))
            cv2.imwrite(filename, frame)
            save_id += 1
            count += 1
            print("Number of saved frames: %d" % (save_id))
            print("--------------------------------------------------------\n")
    cap.release()
    cv2.destroyAllWindows()