from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from model.data import get_val_transforms, collate_fn, VideoDataset
from config.constants import (
    IMG_DIM,
    TEST_BATCH_SIZE,
    MODEL_TYPE,
    MODEL_STATE_DIR,
    NUM_CLASSES,
    SUBMISSION_DIR,
)
from torch.utils.data import DataLoader
import pandas as pd
from data_prep import sv_frame
import os
from model.inference_gndtruth import inference_loop
import time
import numpy as np
import check
import tempfile

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

# from tensorflow import keras
app = Flask(__name__, static_url_path="/static")
# app = Flask(__name__)
# model = ResNet50()


@app.route("/", methods=["GET"])
def hello_word():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    start_time = time.time()
    videofile = request.files["videofile"]
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    # Save the uploaded video file to the temporary directory
    video_path = os.path.join(temp_dir, "uploaded_video.mp4")
    videofile.save(video_path)
    print(video_path)
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
    model_paths = sorted(list(MODEL_STATE_DIR.glob("best_val_loss*")))
    print(model_paths, "model_paths")
    overall_results = dict()
    print(overall_results, "overall_results")

    for m in range(len(model_paths)):
        model_path = model_paths[m]

        overall_results[m] = inference_loop(model, model_path, inf_loader, device)

    for i, model_result in overall_results.items():
        model_folder, model_logits, model_pred, model_target, model_loss = model_result
        print(label_dict[model_pred[0]])
        predictions = pd.Series(label_dict[model_pred[0]], name="prediction")
        predictions.to_csv(SUBMISSION_DIR / "vr-web.txt", sep="\t", header=False)
        model_logits = np.array(model_logits)
        # Use argsort to get the indices of the top 5 values
        top5_indices = model_logits.argsort()[-5:][::-1]
        arr = np.array(model_logits)
        top5_indices = arr.argsort()[0][-5:][::-1]
        top5_elements = arr[0][top5_indices].tolist()
        print(top5_elements, "ele")

        print(model_logits, top5_indices)
        top5_indices = model_logits.argsort(axis=1)[:, -5:]
        top5 = model_logits.argsort(axis=0)[:, -5:]
        # Map the class indices to action labels
        top5_actions = [
            [label_dict[idx] for idx in sample_indices]
            for sample_indices in top5_indices
        ]
        top5_actions = top5_actions[0][::-1]
        print(top5_actions, top5)

    print(videofile.filename)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Predict function execution time: {elapsed_time} seconds")

    # print(top_actions_dict,"top_actions_dict")
    return render_template(
        "index.html",
        prediction=label_dict[model_pred[0]],
        filename=videofile.filename,
        top_5=top5_actions,
        elapsed_time=elapsed_time,
        accuraies_top_5=top5_elements,
    )
    # return render_template('index.html', prediction=classification)


code_has_run = False


@app.route("/", methods=["POST"])
def index():
    check.output()
    return render_template("index.html")


# @app.route('/display_image/')
# def display_image():
#     global code_has_run
#     # print(filename_img,'filename_img')
#     if code_has_run:
#         return "Code has already run"

#     code_has_run = True
#     start_time = time.time()
#     def process_frame(frame, body_estimation):
#         candidate, subset = body_estimation(frame)
#         canvas = util.draw_bodypose(frame, candidate, subset)
#         return canvas

#     body_estimation = Body('model\state_dict\openpose\\body_pose_model.pth')

#     # Specify the directory path where your video files are located
#     directory_path = 'EE6222_data/web_try/Run'

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

#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         middle_frame_index = frame_count // 2
#         current_frame_index = 0
#         output_filename = 'middle_frame.jpg'
#         print("Frame count",frame_count)
#         # Construct the full path to save the middle frame image
#         output_path = os.path.join('Static', output_filename)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             if current_frame_index == middle_frame_index:
#                 processed_frame = process_frame(frame, body_estimation)
#             # Save the processed middle frame as an image
#                 cv2.imwrite(output_path, processed_frame)
#                 break

#             current_frame_index += 1

#             # Write the processed frame to the output video
#             # output_video.write(processed_frame)

#         # Release video objects
#         cap.release()
#         # output_video.release()

#     print("Processing complete. Output videos saved with new names.")


#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Predict function execution time: {elapsed_time} seconds")
#     print(output_filename,'output_filename')

#     # return redirect(url_for('static/img', filename=output_filename),code=301)
#     return send_from_directory('static/', filename_img=output_filename)
#     # return render_template('image_display.html', filename_img=output_filename)
if __name__ == "__main__":
    app.run(port=3000, debug=True)
