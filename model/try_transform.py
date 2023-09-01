import cv2
import numpy as np
import torch 
import glob
import os
import albumentations as a
from pathlib import Path
from config.constants import N_FRAMES, INTERVAL, IMG_DIM, IMG_SHAPE, OPENPOSE_STATE_DIR
from torch.utils.data import Dataset
from .openpose_pytorch.src.body import Body_HM

# Dataset class with shape (T, C, H, W)
class VideoDataset(Dataset):
    def __init__(self, df, transforms, img_path_col="path", labelAvailable=True):    
        self.df = df  
        self.transforms = transforms
        self.img_path_col = img_path_col
        self.labelAvailable = labelAvailable

    def __len__(self):
        print("here 1")
        return len(self.df)

    def __getitem__(self, idx):

        # Video path
        img_folder = self.df[self.img_path_col].iloc[idx] + "/*.jpg"
        print(img_folder)
        # Append path of all frames in a video
        all_img_path = glob.glob(img_folder)
        all_img_path = sorted(all_img_path)
        print(img_folder)
        v_len = len(all_img_path)
        print(v_len)
        # Uniformly samples N_FRAMES number of frames   
        '''
        Here is where the selected number of frames are selected
        '''
        if v_len > N_FRAMES*INTERVAL:
            print(N_FRAMES*INTERVAL)
            frame_list = np.arange(np.int64((v_len - (N_FRAMES*INTERVAL))*0.5), np.int64((v_len - (N_FRAMES*INTERVAL))*0.5) + N_FRAMES*INTERVAL, INTERVAL)
        else:
            frame_list = np.arange(0, N_FRAMES*INTERVAL, INTERVAL)
        print("here 2")
        img_path = []
        for fn in range(v_len):
            if (fn in frame_list):
                img_path.append(all_img_path[fn])
        print(img_path)        
        images = []
        for p2i in img_path:
            p2i = Path(p2i)
            img = cv2.imread(p2i.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        print("here 3")
        while (len(images) < N_FRAMES):
            '''If the size is not enough black sreen is added for the rest of the frames'''
            images.append(np.zeros((IMG_DIM,IMG_DIM,3), np.uint8))
        images_tr = []

        body_estimation_hm = Body_HM(OPENPOSE_STATE_DIR)

        for image in images:
            print("here 4")
            if len(images_tr) == 0:
                augmented = self.transforms(image=image)
                image = augmented['image']
                body_heatmap = body_estimation_hm(image)
                hm = torch.Tensor((1 - body_heatmap[:,:,-1]))
                hm = torch.stack((hm,hm,hm), dim = 0)
                images_tr.append(torch.Tensor(hm))
                data_replay = augmented['replay']
                
            else:
                image = a.ReplayCompose.replay(data_replay, image=image)
                image = image['image']
                body_heatmap = body_estimation_hm(image)
                hm = torch.Tensor((1 - body_heatmap[:,:,-1]))
                hm = torch.stack((hm,hm,hm), dim = 0)
                images_tr.append(torch.Tensor(hm))

        if len(images_tr)>0:
            images_tr = torch.stack(images_tr)   
        print("here 5")
        if self.labelAvailable == True:
            # Label
            label = self.df["label"].iloc[idx]
            return img_folder, images_tr, label
        else:
            return img_folder, images_tr

def get_train_transforms(img_dim):
    trans = a.ReplayCompose([
        a.CLAHE(clip_limit=(10, 10), tile_grid_size=(8, 8), always_apply=True),
        a.RandomGamma((150, 150), always_apply=True),
        a.PadIfNeeded(img_dim, img_dim),
        a.CenterCrop(img_dim, img_dim, always_apply=True),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Rotate(limit=120, p=0.8),

    ])
    print("transfered")
    return trans

def get_val_transforms(img_dim):
    trans = a.ReplayCompose([
        a.CLAHE(clip_limit=(10, 10), tile_grid_size=(8, 8), always_apply=True),
        a.RandomGamma((150, 150), always_apply=True),
        a.PadIfNeeded(img_dim, img_dim),
        a.CenterCrop(img_dim, img_dim, always_apply=True),
    ])
    # print("transformed")
    return trans

def collate_fn(batch):
    if len(batch[0]) == 3:
        print("collate-1")
        img_folder_batch, imgs_batch, label_batch = list(zip(*batch))
        label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
        labels_tensor = torch.stack(label_batch)
    else:
        print("collate-2")
        img_folder_batch, imgs_batch = list(zip(*batch))   
    print("collate-3")   
    img_folder_batch = [folders for folders in img_folder_batch if len(folders)>0]
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    
    #imgs_tensor = torch.stack(torch.Tensor(imgs_batch))
    print("collate-4")
    if IMG_SHAPE == 'BCTHW':
        print("if")
        imgs_batch = [torch.transpose(imgs, 1, 0) for imgs in imgs_batch if len(imgs)>0]
    elif IMG_SHAPE == 'BTCHW':
        imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    print("collate-5")
    imgs_tensor = torch.stack(imgs_batch)
    
    if len(batch[0]) == 3:
        print("collate-6")
        return img_folder_batch, imgs_tensor,labels_tensor
    else:
        return img_folder_batch, imgs_tensor
    

