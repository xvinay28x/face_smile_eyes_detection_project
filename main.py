from flask import Flask, render_template, request, Response
from camera import Video
import cv2

app = Flask(__name__)

def face_gen(camera):  
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def smile_gen(camera):  
    while True:
        frame = camera.smile_detect()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def eyes_gen(camera):  
    while True:
        frame = camera.eyes_detect()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')                


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/face_feed')
def face_feed():
    return Response(face_gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/smile_feed')
def smile_feed():
    return Response(smile_gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/eyes_feed')
def eyes_feed():
    return Response(eyes_gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/face", methods=["POST"])
def face():
    return render_template("face.html")


@app.route("/smile", methods=["POST"])
def smile():
    return render_template("smile.html")


@app.route("/eyes", methods=["POST"])
def eyes():
    return render_template("eyes.html")  


if __name__ == "__main__": 
    app.run(debug=True)    