import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from detect import detect


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
path = "./uploaded_videos"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)


app = Flask(__name__)
UPLOAD_FOLDER = "./uploaded_videos"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Check whether the specified path exists or not


@app.route("/upload", methods=["GET"])
def upload():
    return render_template("upload.html")


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        filename = secure_filename(f.filename)
        if filename[-3:] != "mp4":
            return "<h1>Video should be with mp4 extension"
        # f.save(filename)
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return render_template("uploader.html", video=filename, result=detect())


@app.route("/", methods=["GET"])
def hello():
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
