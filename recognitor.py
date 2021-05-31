import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
from urllib.request import urlopen
import numpy as np
from keras.models import model_from_json
import sys
import math
from PIL import Image
import argparse
import mysql.connector
from mysql.connector import MySQLConnection, Error

# Define the parser
parser = argparse.ArgumentParser(description='Input args')
parser.add_argument('--url', action="store", dest='url', default='')
parser.add_argument('--cam_id', action="store", dest='cam_id', default=0)
parser.add_argument('--img_size', action="store", dest='img_size', default=64)
parser.add_argument('--model_name', action="store", dest='model_name', default='VGG16')
args = parser.parse_args()

print('\n\n')
print('====================/Arguments/====================')
def connect():
    """ Connect to MySQL database """
    conn = None
    try:
        conn = mysql.connector.connect(host='localhost',
                                       database='EmployeeManagement',
                                       user='root',
                                       password='password')
        if conn.is_connected():
            print('Connected to MySQL database')
            return conn

    except Error as e:
        print(e)

connector = connect()
cursor=connector.cursor()

url = args.url
print('Video source: ' + url)
cam_id=int(args.cam_id)
print('Camera id: ' + str(cam_id))
img_size=int(args.img_size)
print('Image size: ' + str(img_size))
faceCascPath = 'models/haarcascade_frontalface_default.xml'
print('Face detector: ' + faceCascPath)
model_name= args.model_name

training_sequence_file = open("models/" + model_name + "/training_sequence.txt", "r")
ids=temp = training_sequence_file.read().splitlines()
training_sequence_file.close();
num_classes=len(ids)
print('Number of trained classes: ' + str(num_classes))
print('====================/Arguments/====================')
print('\n')

print('====================/Recognition/====================')
# load json and create model
json_file = open('models/' + model_name + '/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/" + model_name +"/model.h5")
print("Loaded model from disk")

faceCascade = cv2.CascadeClassifier(faceCascPath)
if(url==''):
    video_capture = cv2.VideoCapture(cam_id)
else:
    video_capture = cv2.VideoCapture(url)

def resample_image(img):
    img = Image.fromarray(img.astype('uint8'), 'L')
    img = img.resize((img_size,img_size), Image.ANTIALIAS)
    return np.array(img)

# recognize on live video stream
# loop over the frames from the video stream
count=0;
res = np.zeros(num_classes)
person_name=''
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert current frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in current frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64,64))

    # Loop through faces which are detected
    for face in faces:
        # Grayscale, reshape the input before prediction 
        face_x, face_y, face_w, face_h = face
        face_gray = gray[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
        pred_img = Image.fromarray(face_gray)
        pred_img= np.array(pred_img)
        pred_img=resample_image(pred_img)
        pred_img=pred_img.reshape(-1, img_size, img_size, 1).astype('float32') / 255.0
        Y_pred =  loaded_model.predict(pred_img)
        result = int(np.argmax(Y_pred, axis=1).astype('int'))
        person_id = ids[result]
        spargs=[person_id]
        cursor.callproc('Proc_GetEmployeeName', spargs)
        #print out the result
        query_res=[]
        for result in cursor.stored_results():
            query_res= result.fetchall()
            break;
        person_name = [x[0] for x in query_res]
        print('Recognized id: ' + person_id + ', person name: ' + str(person_name))

        # count+=1;

        # res[result]+=float(Y_pred[0][result])
        # if count==5:
        #     person_id = ids[np.argmax(res)]
        #     spargs=[person_id]
        #     cursor.callproc('Proc_GetEmployeeName', spargs)
        #     # print out the result
        #     query_res=[]
        #     for result in cursor.stored_results():
        #         query_res= result.fetchall()
        #         break;
        #     person_name = [x[0] for x in query_res]
        #     count=0
        #     res=np.zeros(num_classes)
        #     print('Recognized id: ' + person_id + ', person name: ' + str(person_name))

        # draw a rectangle around the face
        cv2.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), (0,0,255), 2)
        # draw person name
        cv2.putText(frame, str(person_name), (face_x, face_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print('====================/Recognition/====================')
print('\n\n')
