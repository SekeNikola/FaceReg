'''Face Detection Login App '''

import random
import face_recognition
import cv2
import glob
from flask import Flask, render_template, redirect, url_for
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/registered')
def registered():
    return render_template('registered.html')


@app.route('/reject')
def reject():
    return render_template('reject.html')

@app.route('/login', methods=["GET", "POST"])
def login():

    page_name = 'reject'

    video_capture = cv2.VideoCapture(0)
    # Load faces
    faces = 'faces/*.jpg*'
    face = glob.glob(faces)
    print(face)
    for fn in face:
        try_image = face_recognition.load_image_file(f'{fn}')
        try_face_encoding = face_recognition.face_encodings(try_image)

    if not try_face_encoding:
        print("No face found on the image")
        return redirect(url_for(page_name))

    try_face_encoding = try_face_encoding[0]

    # # Array of faces
    known_face_encodings = [
        try_face_encoding,
    ]

    face_locations = []
    face_encodings = []
    process_this_frame = True

    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)

            if True in matches:
                first_match_index = matches.index(True)
                page_name = 'dashboard'
                break

    # if user is NOT found release the capture and redirect
    video_capture.release()
    cv2.destroyAllWindows()

    return redirect(url_for(page_name))


# Register
@app.route('/register', methods=["GET", "POST"])
def register():
    video_capture = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while(True):
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if w <= 200:
                x = 0
                y = 20
                text_color = (0, 255, 0)
                cv2.putText(
                    frame, "Please get closer", (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1
                )
            else:
                x = 0
                y = 20
                text_color = (0, 255, 0)
                cv2.putText(
                    frame, "Press q to take image", (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=1
                )

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            image_name = str(random.randint(1, 100))
            cv2.imwrite(f'faces/try{image_name}.jpg', frame)
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return redirect(url_for('registered'))


if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()