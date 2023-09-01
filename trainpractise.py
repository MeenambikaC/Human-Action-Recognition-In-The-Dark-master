import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from config.constants import (TRAIN_VID_DIR, TRAIN_VID_FOLDER, TRAIN_IMG_FOLDER, VAL_VID_DIR, VAL_VID_FOLDER, VAL_IMG_FOLDER, 
                            MAP_TABLE_DIR, TRAIN_CSV, TRAIN_IMG_DIR, EXTENSION, VAL_CSV, VAL_IMG_DIR,
                            IMG_DIM, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, MODEL_TYPE,
                            MODEL_NAME, DROPOUT, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, LR, ACCUM_ITER, 
                            NUM_WARMUP_EPOCHS, NUM_EPOCHS, NUM_COS_CYCLE, MODEL_STATE_DIR, TOP_K, NUM_CLASSES, CHECKPOINT_DIR, SUBMISSION_DIR)
from model.try_transform import get_train_transforms, collate_fn, VideoDataset
from transformers import get_cosine_schedule_with_warmup
from model.model import HARModel
from model.train import train_loop, val_loop
from data_prep import sv_frame
from torchvision.models.video import r3d_18, r2plus1d_18, R3D_18_Weights, R2Plus1D_18_Weights
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os

DATA_DIR  = Path('EE6222_data')
Try_VID_FOLDER = 'try_train'
Try_IMG_FOLDER = 'try_train_img'
Try_VID_DIR = DATA_DIR / Try_VID_FOLDER
Try_IMG_DIR = DATA_DIR / Try_IMG_FOLDER
Try_CSV = DATA_DIR / 'try_train.txt'

sv_frame(Try_VID_DIR, Try_VID_FOLDER, Try_IMG_FOLDER)
df_train = pd.read_csv(Try_CSV, sep="\t", header=None, index_col=0)
df_train.columns = ['label', 'path']
print(df_train)
print("first")
df_train['path'] = str(Try_IMG_DIR) + '/' + df_train['path'].str.replace(EXTENSION, "")
# print(df_train)
# df_train = df_train.iloc[:10]
train_transforms = get_train_transforms(IMG_DIM)
print('before')
train_data = VideoDataset(df=df_train,
                          transforms=train_transforms, 
                          labelAvailable=True)
print("here")
print(df_train)

train_loader = DataLoader(train_data, batch_size= TRAIN_BATCH_SIZE,
                    shuffle=True, collate_fn= collate_fn)

fold_results = dict()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if MODEL_TYPE == 'cnn-rnn':
    model = HARModel(
        model_name=MODEL_NAME,
        dropout=DROPOUT,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        rnn_num_layers=RNN_NUM_LAYERS,
        num_classes=NUM_CLASSES,
        pretrained=True)
elif MODEL_TYPE == 'r3d_18':
    model = r3d_18(weights=R3D_18_Weights.DEFAULT, progress=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
elif MODEL_TYPE == 'r2plus1d_18':
    print("MODEL_TYPE")
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT, progress=False)
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, NUM_CLASSES)

model = model.to(device)
print("CUDA")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# Configs for scheduler
print(len(train_loader),'len(train_loader)')
print(ACCUM_ITER,'ACCUM_ITER')
num_steps_per_epoch = np.round_(len(train_loader)/ACCUM_ITER)
print(num_steps_per_epoch,'num_steps_per_epoch')
num_warmup_steps = NUM_WARMUP_EPOCHS * num_steps_per_epoch
print(num_warmup_steps,'num_warmup_steps')
num_training_steps = NUM_EPOCHS * num_steps_per_epoch
print(num_training_steps,'num_training_steps')
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=NUM_COS_CYCLE
)

best_accuracy = 0
best_val_loss = 999_999
fold_train_loss = []
fold_val_loss = []
fold_accuracy = []
fold_k_score = []

for epoch_num in range(NUM_EPOCHS):
    lr_list = []
    epoch_info = f'Epoch {epoch_num+1}/{NUM_EPOCHS}'
    print(epoch_info,'epoch_info')
    lr_list, train_losses = train_loop(
        model, train_loader, device, optimizer, ACCUM_ITER, 
        scheduler=scheduler,epoch_info=epoch_info
    )

    # val_losses, score, top_k_score = val_loop(
    #     model, val_loader, device, label_list=label_list, k=TOP_K, epoch_info=epoch_info
    # )

    avg_train_loss = np.mean(train_losses)
    print(avg_train_loss,'avg_train_loss')
    # avg_val_loss = np.mean(val_losses)
    #  avg_val_accuracy_score = np.mean(score)
    # avg_val_top_k_accuracy = np.mean(top_k_score)
    
    fold_train_loss.append(avg_train_loss)
    print(avg_train_loss,'avg_train_loss')
    # fold_val_loss.append(avg_val_loss)
    # fold_accuracy.append(avg_val_accuracy_score)
    # fold_k_score.append(avg_val_top_k_accuracy)

    # Save model if val loss improves
    # if avg_val_loss < best_val_loss:
    #     model_state_dict = model.state_dict()
        
    #     model_name = f'best_val_loss.pt'
    #     torch.save(
    #         model.state_dict(),
    #         MODEL_STATE_DIR / model_name
    #     )
        
    #     best_val_loss = avg_val_loss

    # Save checkpoint at the last 
    if epoch_num == NUM_EPOCHS - 1:
        model_name = f'model_checkpoint.pt'
        torch.save({'epoch':epoch_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_DIR / model_name)

fold_results[0] = (fold_train_loss, fold_accuracy, fold_k_score)

(train_loss, accuracy, k_score) = fold_results[0]
print(fold_results[0],'fold_results[0]')
# plt.figure(figsize=(5, 5))
# plt.plot(train_loss, label='Train')
# plt.plot(val_loss, label='Val')
# plt.legend(loc='upper right')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title(f'Training Loss and Validation Loss')
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.plot(fold_accuracy, label='Accuracy')
# plt.legend(loc='upper right')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title(f'Validation Accuracy')
# plt.show()

