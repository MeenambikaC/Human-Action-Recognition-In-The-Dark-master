import os
import utils
from config.constants import (TRAIN_VID_FOLDER, TRAIN_IMG_FOLDER, TRAIN_VID_DIR, 
                                VAL_VID_FOLDER, VAL_IMG_FOLDER, VAL_VID_DIR, 
                                EXTENSION, N_FRAMES)

listOfCategories = os.listdir(TRAIN_VID_DIR)

def list_cat(train_path):
    for cat in listOfCategories:
        print("category:", cat)
        path2acat = os.path.join(train_path, cat)
        listOfSubs = os.listdir(path2acat)
        print("number of sub-folders:", len(listOfSubs))
        print("-"*50)
    return 

def sv_frame(path, subfolder, subfolder_jpg):
    for root, dirs, files in os.walk(path, topdown=False):
        #print(f'root: {root}')
        for name in files:
            if EXTENSION not in name:
                continue
            path2vid = os.path.join(root, name)
            #print(path2vid)
            frames, vlen = utils.get_frames(path2vid, n_frames= N_FRAMES)
            path2store = path2vid.replace(subfolder, subfolder_jpg)
            path2store = path2store.replace(EXTENSION, "")
            print(path2store)
            os.makedirs(path2store, exist_ok= True)
            utils.store_frames(frames, path2store)
        print("-"*50)  

#sv_frame(TRAIN_VID_DIR, TRAIN_VID_FOLDER, TRAIN_IMG_FOLDER)
#sv_frame(VAL_VID_DIR, VAL_VID_FOLDER, VAL_IMG_FOLDER)