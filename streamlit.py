import streamlit as st
import os
import new
import time
import check

# Constants
UPLOAD_DIRECTORY = "temp"


def save_uploaded_file(uploaded_file):
    try:
        path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return path
    except Exception as e:
        st.write(e)
        return None


st.title("Video Action Classifier")

uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "mkv", "avi", "mov"]
)
if uploaded_file is not None:
    st.write("File successfully uploaded.")

    with st.spinner("Predicting... Please wait."):
        video_path = save_uploaded_file(uploaded_file)
        print(video_path)
        if video_path:
            start_time = time.time()
            class_belong, val = new.predict_single_action(video_path, 11)
            end_time = time.time()
            elapsed_time = end_time - start_time

            st.write(f"File Name: {uploaded_file.name}")

            # Display predicted action in a larger font size
            st.markdown(
                f"<center><p>Action Prediction</p><h1 style='text-align: center; color: red;'>{class_belong}</h1></center><br><br>",
                unsafe_allow_html=True,
            )
            # Display predicted action in a larger font size
            st.markdown(
                f"<center><p>Confidence</p><h4 style='text-align: center; color: red;'>{round(val*100,2)}%</h4></center>",
                unsafe_allow_html=True,
            )
            st.write(f"Time delay for the prediction: {elapsed_time:.2f} seconds")

            # You may need additional logic to display the 'middle_frame.jpg' as well
            image_path = os.path.join("temp", "middle_frame.jpg")

            # Check if the middle_frame.jpg exists and delete it if it does
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted existing file: {image_path}")

            time_taken = check.output(uploaded_file.name)

            if os.path.exists(image_path):
                st.image(image_path, caption="Processed Frame", use_column_width=True)
                st.write(f"Time to generate frame: {time_taken} seconds")

            video_path = f"{uploaded_file.name}_output.mp4"

            # Check if the middle_frame.jpg exists and delete it if it does
            if os.path.exists("temp/" + video_path):
                os.remove(video_path)
                print(f"Deleted existing file: {video_path}")

            time_taken = check.output(video_path)

            if os.path.exists("temp/" + video_path):
                st.video(
                    "temp/" + video_path,
                    caption="Processed Video",
                    use_column_width=True,
                )
                st.write(f"Time to generate video: {time_taken} seconds")


# Function to create a gauge chart
def render_gauge(score):
    option = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "Accuracy",
                "type": "gauge",
                "detail": {"formatter": "{value}%"},
                "data": [{"value": score, "name": "Accuracy Score"}],
            }
        ],
    }
    st_echarts(options=option, height="200px")
