import cv2 as cv
import numpy as np
import os
import sqlite3
from PIL import Image

def cam_test():
    detector  =cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv.VideoCapture(0)
    student_id = input('\n enter your id:')
    student_name= input('\n enter your name:')
    students_details= (student_id,student_name)
    con = sqlite3.connect('students.db')
    cursor= con.cursor()


    cursor.execute("CREATE TABLE if not exists students(id, name)")
    cursor.execute("INSERT INTO students VALUES (?,?)" ,  students_details)
    con.commit()
    count = 0

    while True:
         
        ret, img = cam.read ()
        gray  = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
        )

        for (x,y,w,h) in faces:
            cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0,), 2)
            count += 1
            cv.imwrite('dataset/student.'+str(student_id)+'.'+str(count)+'.jpg', gray[y:y+h, x:x+w])
        cv.imshow('Camera', img)

        key = cv.waitKey(20) & 0xFF
        if key ==27:
            break
        elif count == 30:
            break
    cam.release()
    cv.destroyAllWindows()
# cam_test() # function calling 


def train():
    data_path ='dataset'
    detector =cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    recorgniser = cv.face.LBPHFaceRecognizer_create()


    def getImageData(data_path):
        images = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        samples = []
        students_ids =[]
        
        for image in images:
            pillow_image = Image.open(image).convert('L')
            numpy_image =np.array(pillow_image, 'uint8')

            student_id =int(os.path.split(image)[-1].split('.')[1])
            faces = detector.detectMultiScale(numpy_image)
            for (x,y,w,h) in faces:
                samples.append(numpy_image[y:y+h, x:x+w])
                students_ids.append(student_id)

        return samples, students_ids
        
    print('Training.......') 
    all_faces, all_ids =getImageData(data_path)
    recorgniser.train( all_faces, np.array(all_ids))   
    recorgniser.write('trainer/trainer.yml')
# train()


def old_train():
    path = "dataset"
    detector = cv.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    recorginser = cv.face.LBPHFaceRecognizer_create()
    
    def getImages(path):
        images = [os.path.join(path, f) for f in os.listdir(path) ]
        samples = []
        ids = []
        for image in images:
            pil_image = Image.open(image).convert('L')
            num_image = np.array(pil_image, "uint8")
            id = int(os.path.split(image)[-1].split(".")[1])
            det_faces = detector.detectMultiScale(num_image)

            for (x,y,w,h) in det_faces:
                samples.append(num_image[y:y+h, x:x+w])
                ids.append(id)
        return samples, ids
        
    faces, ids = getImages(path)
    recorginser.train(faces, np.array(ids))
    recorginser.write('trainer/trainer.yml')
    return 'Finished Trained our model'
# train()

def testing():
    detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    recorginser = cv.face.LBPHFaceRecognizer_create()
    recorginser.read('trainer/trainer.yml')
    font = cv.FONT_HERSHEY_SIMPLEX
    id = 0

    cam = cv.VideoCapture(0)

    conn = sqlite3.connect('students.db')

    cur = conn.cursor()
    students = cur.execute("SELECT * FROM students")
    students_names = ['None']

    for student in students:
        students_names.append(student[1])
        print(student)


    while True:
        frame, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            minNeighbors=5,
            scaleFactor=1.2,
        )
        for (x,y,h,w) in faces:
            cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
            id, confidence = recorginser.predict(gray[y:y+h, x:x+w])
            
            if confidence < 100 > 40:
                id = students_names[id]
                # success(id)
                confidence = "{0}%".format(round(100 - confidence))
            else:
                id = "Unkown Student"
                confidence = "{0}%".format(round(100 - confidence))
        
            cv.putText(
                img,
                id,
                (x+5, y-5),
                font,
                1,
                (225,0,0),
                2
            )

            cv.putText(
                img,
                confidence,
                (x-60, y-5),
                font,
                1,
                (225,0,0),
                2
            )

        cv.imshow('Camera', img)
        key = cv.waitKey(20) & 0xFF
        if key == 27:
            break
    cam.release()
    cv.destroyAllWindows()

testing()
# success(id)





            