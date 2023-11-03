from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    url_for,
    redirect,
)
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
import torchvision.transforms as transforms
from model.data import get_val_transforms, collate_fn, VideoDataset
from config.constants import (
    IMG_DIM,
    TEST_BATCH_SIZE,
    MODEL_TYPE,
    MODEL_STATE_DIR,
    NUM_CLASSES,
    TOP_K,
    SUBMISSION_DIR,
)
from torch.utils.data import DataLoader
import pandas as pd
from data_prep import sv_frame
import os
import cv2
from model.inference_gndtruth import inference_loop
from model.openpose_pytorch.src import util
from model.openpose_pytorch.src.model import bodypose_model
from model.openpose_pytorch.src.body import Body
from model.openpose_pytorch.src.util import padRightDownCorner
import time
import numpy as np


def output(filename):
    def process_frame(frame, body_estimation):
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(frame, candidate, subset)
        return canvas

    start_time = time.time()
    body_estimation = Body("model/state_dict/openpose/body_pose_model.pth")

    # Specify the directory path where your video files are located
    directory_path = "temp/"

    # List all files in the specified directory
    file_list = os.listdir(directory_path)

    # Iterate through the list of files

    # Construct the full path to the video file
    video_path = os.path.join(directory_path, filename)

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = frame_count // 2
    current_frame_index = 0
    output_filename = "middle_frame.jpg"
    print("Frame count", frame_count)
    # Construct the full path to save the middle frame image
    output_path = os.path.join("temp", output_filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame_index == middle_frame_index:
            processed_frame = process_frame(frame, body_estimation)
            # Save the processed middle frame as an image
            cv2.imwrite(output_path, processed_frame)
            break

        current_frame_index += 1

        # Write the processed frame to the output video
        # output_video.write(processed_frame)

    # Release video objects
    cap.release()
    # output_video.release()

    print("Processing complete. Output videos saved with new names.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Predict function execution time: {elapsed_time} seconds")
    print(output_filename, "output_filename")
