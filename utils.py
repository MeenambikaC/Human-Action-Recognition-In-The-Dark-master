import cv2
import os
import numpy as np
import torch 
import PIL
import matplotlib.pyplot as plt

def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    ids = []
    labels = []
    for catg in listOfCats:
        path2catg = os.path.join(path2ajpgs, catg)
        listOfSubCats = os.listdir(path2catg)
        path2subCats= [os.path.join(path2catg,los) for los in listOfSubCats]
        ids.extend(path2subCats)
        labels.extend([catg]*len(listOfSubCats))
    return ids, labels, listOfCats 

def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        #if (fn in frame_list):
        '''
        Need to think of the use of n_frames
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frames.append(frame)
    v_cap.release()
    return frames, v_len

def store_frames(frames, path2store):
    for ii, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        path2img = os.path.join(path2store, "frame"+str(ii).zfill(3)+".jpg")
        cv2.imwrite(path2img, frame)

def show_img_from_dataset(dataset, index):
    for i, (images, label) in enumerate(dataset):
        if i == index:
            print(f'image shape: {images.shape}')
            print(f'label: {label}')
            for i in range(len(images)):
                plt.imshow((images[i].permute(1, 2, 0) * 255).numpy().astype(np.uint8))
                plt.show()