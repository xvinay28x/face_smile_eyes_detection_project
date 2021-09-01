import cv2

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_detect = cv2.CascadeClassifier("haarcascade_smile.xml")

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()   

    def get_frame(self):
        ret,frame = self.video.read()
        faces = face_detect.detectMultiScale(frame, 1.3, 1)
        for x,y,w,h in faces:
            x1,y1 = x+w, y+h
            cv2.rectangle(frame, (x,y), (x1, y1), (5, 165, 201),3)
        ret,jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()   
    def smile_detect(self):
        ret,frame = self.video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(5, 165, 201),3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            smile = smile_detect.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in smile:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)  
        ret,jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()
    def eyes_detect(self):
        ret,frame = self.video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(5, 165, 201),3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_detect.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255, 204, 51),3)
        ret,jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()