
''' My  Try'''
import os
import cv2
import util
# from model import bodypose_model
# from body import Body
from model.openpose_pytorch.src import util
# import util
from model.openpose_pytorch.src.model import bodypose_model
from model.openpose_pytorch.src.body import Body

def process_frame(frame, body_estimation):
    candidate, subset = body_estimation(frame)
    canvas = util.draw_bodypose(frame, candidate, subset)
    return canvas

if __name__ == "__main__":
    # Initialize the pose estimation model
    body_estimation = Body('body_pose_model.pth')

    # Specify the directory path where your video files are located
    directory_path = 'D:/Fifth Semester/DSE Project/frontEnd/server-backend/public/Images'

    # List all files in the specified directory
    file_list = os.listdir(directory_path)

    # Iterate through the list of files
    for filename in file_list:
        # Construct the full path to the video file
        video_path = os.path.join(directory_path, filename)

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec for MP4 and create a VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'H264' for MP4
        input_video_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_video_name = f'{input_video_basename}_output.mp4'
        print(f"Processing {filename} => {output_video_name}")
        output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, body_estimation)

            # Write the processed frame to the output video
            output_video.write(processed_frame)

        # Release video objects
        cap.release()
        output_video.release()

    print("Processing complete. Output videos saved with new names.")

