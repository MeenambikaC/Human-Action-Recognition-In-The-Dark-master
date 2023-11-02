from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    url_for,
    redirect,
)
import pandas as pd
import numpy as np
import os
import new
import time
import check

app = Flask(__name__, static_url_path="/static")


@app.route("/", methods=["GET"])
def hello_word():
    return render_template("newindex.html")


@app.route("/", methods=["POST"])
def predict():
    start_time = time.time()
    videofileold = request.files["videofile"]
    video_path = "temp/" + videofileold.filename

    if os.path.exists(video_path):
        os.remove(video_path)
    videofile = request.files["videofile"]
    video_path = "temp/" + videofile.filename
    # print(video_path)
    videofile.save(video_path)
    class_belong, val = new.predict_single_action(video_path, 11)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Predict function execution time: {elapsed_time} seconds")
    check.output()
    return render_template(
        "newindex.html",
        filename=videofile.filename,
        class_belong=class_belong,
        val=val,
        elapsed_time=elapsed_time,
    )


if __name__ == "__main__":
    app.run(port=3003, debug=True)
