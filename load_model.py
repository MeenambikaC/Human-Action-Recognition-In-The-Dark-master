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
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix

# from tensorflow import keras
print("hi")
sv_frame("EE6222_data/web_try", "web_try", "web_img_try")
df_test = pd.read_csv("EE6222_data/web_try.txt", sep="\t", header=None, index_col=0)
df_test.columns = ["label", "path"]
df_test["path"] = (
    str("EE6222_data/web_img_try") + "/" + df_test["path"].str.replace(".mp4", "")
)
print(df_test.head())

inf_transforms = get_val_transforms(IMG_DIM)
inf_data = VideoDataset(df=df_test, transforms=inf_transforms, labelAvailable=True)
inf_loader = DataLoader(
    inf_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if MODEL_TYPE == "r2plus1d_18":
    print("r2plus1d_18", "here")
    model = r2plus1d_18(weights=None, progress=False)
    num_features = model.fc.in_features
    print(num_features, "Num_of_features")
    model.fc = nn.Linear(num_features, NUM_CLASSES)
# print(MODEL_STATE_DIR,'MODEL_STATE_DIR')
# model_paths = sorted(list(MODEL_STATE_DIR.glob('be8ab2*')))
label_dict = {
    0: "Drink",
    1: "Jump",
    2: "Pick",
    3: "Pour",
    4: "Push",
    5: "Run",
    6: "Sit",
    7: "Stand",
    8: "Turn",
    9: "Walk",
    10: "Wave",
}
model_paths = sorted(list(MODEL_STATE_DIR.glob("best_val_loss*")))
print(model_paths, "model_paths")
overall_results = dict()
print(overall_results, "overall_results")


for m in range(len(model_paths)):
    model_path = model_paths[m]

    overall_results[m] = inference_loop(model, model_path, inf_loader, device)


for i, model_result in overall_results.items():
    model_folder, model_logits, model_pred, model_target, model_loss = model_result
    # label_dict={0: 'Drink', 1: 'Jump', 2: 'Pick', 3: 'Pour', 4: 'Push', 5: 'Run', 6: 'Sit', 7: 'Stand', 8: 'Turn', 9: 'Walk',10:'Wave'}
    label_dict = {
        0: "Drink",
        1: "Jump",
        2: "Pick",
        3: "Pour",
        4: "Push",
        5: "Run",
        6: "Sit",
        7: "Stand",
        8: "Turn",
        9: "Walk",
        10: "Wave",
    }
    print(label_dict[model_pred[0]])
    model_logits = np.array(model_logits)

    predictions = pd.Series(label_dict[model_pred[0]], name="prediction")
    # predictions.to_csv(SUBMISSION_DIR / 'vr-1.txt', sep='\t', header=False)
    # model_top5_score = top_k_accuracy_score(model_target, model_logits, k=5)

    # Get the top 5 predicted class indices for each sample
    top5_indices = model_logits.argsort(axis=1)[:, -5:]

    # Map the class indices to action labels
    top5_actions = [
        [label_dict[idx] for idx in sample_indices] for sample_indices in top5_indices
    ]

    # print(f'Model Top-5 accuracy score: {model_top5_score}')
    print(f"Top-5 predicted actions:")
    for actions in top5_actions:
        print(actions)


# def process_frame(frame, body_estimation):
#     candidate, subset = body_estimation(frame)
#     canvas = util.draw_bodypose(frame, candidate, subset)
#     return canvas

# if __name__ == "__main__":
#     # Initialize the pose estimation model
#     body_estimation = Body('model\state_dict\openpose\\body_pose_model.pth')

#     # Specify the directory path where your video files are located
#     directory_path = 'EE6222_data/web_try/push'

#     # List all files in the specified directory
#     file_list = os.listdir(directory_path)

#     # Iterate through the list of files
#     for filename in file_list:
#         # Construct the full path to the video file
#         video_path = os.path.join(directory_path, filename)

#         # Get video properties
#         cap = cv2.VideoCapture(video_path)
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Define the codec for MP4 and create a VideoWriter object to save the output video
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'H264' for MP4
#         input_video_basename = os.path.splitext(os.path.basename(video_path))[0]
#         output_video_name = f'{input_video_basename}_output.mp4'
#         print(f"Processing {filename} => {output_video_name}")
#         output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             processed_frame = process_frame(frame, body_estimation)

#             # Write the processed frame to the output video
#             output_video.write(processed_frame)

#         # Release video objects
#         cap.release()
#         output_video.release()

#     print("Processing complete. Output videos saved with new names.")


# # Define the configuration for the model
# MODEL_NAME = "r2plus1d_18"  # You can change this if needed
# DROPOUT = 0.5  # Specify the dropout rate
# RNN_HIDDEN_SIZE = 256  # Specify the RNN hidden size
# RNN_NUM_LAYERS = 2  # Specify the number of RNN layers
# NUM_CLASSES = 11  # Adjust the number of classes as per your use case
# PRETRAINED = True  # Set this to True or False based on your use case

# # Create an instance of the r2plus1d_18 model
# model = r2plus1d_18(pretrained=PRETRAINED)

# # Modify the final fully connected layer for the specified number of classes
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, NUM_CLASSES)

# # Load the model's state dictionary
# model.load_state_dict(torch.load('model/state_dict/bestloss/best_val_loss.pt'))

# # Put the model in evaluation mode (if it's for inference)
# model.eval()
