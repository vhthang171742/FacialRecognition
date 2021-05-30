import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
from urllib.request import urlopen
import numpy as np
from keras.models import model_from_json
from PIL import Image
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Input args')
parser.add_argument('--person_id', action="store", dest='person_id', default='unlabeled')
parser.add_argument('--url', action="store", dest='url', default='')
parser.add_argument('--img_source', action="store", dest='img_source', default='')
parser.add_argument('--cam_id', action="store", dest='cam_id', default=0)
parser.add_argument('--count_start', action="store", dest='count_start', default=0)
parser.add_argument('--target_num', action="store", dest='target_num', default=100)
parser.add_argument('--frame_step', action="store", dest='frame_step', default=1)
parser.add_argument('--color_channels', action="store", dest='color_channels', default=3)
args = parser.parse_args()

# Arguments
print('\n\n')
print('====================/Arguments/====================')
person_id=args.person_id
print('Generating training data for ' + person_id)
url = args.url
print('Video source: ' + url)
img_source = args.img_source
print('Static image source: ' + img_source)
cam_id = args.cam_id
print('Camera id: ' + str(cam_id))
target_num= int(args.target_num)
print('Target image number to generate: ' + str(target_num))
count_start = int(args.count_start)
print('Start of images name sequence: ' + str(count_start))
frame_step=int(args.frame_step)
print('Step between actual input frame: ' + str(frame_step))
color_channels = int(args.color_channels)
print('Color channels: ' + str(color_channels))
print('====================/Arguments/====================')
print('\n')

# cascader
faceCascPath = 'models/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(faceCascPath)

if img_source == '':
    print('====================/Getting faces from video source/====================')
    # output path
    output_path = os.path.join('training_data/images/',(person_id + '/'))
    if not os.path.isdir(output_path):
            os.mkdir(output_path)

    # video reader
    if(url==''):
        video_capture = cv2.VideoCapture(cam_id)
    else:
        video_capture = cv2.VideoCapture(url)

    # recognize on live video stream or video
    # loop over the frames from the video stream/video
    count=count_start;
    success = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if count%frame_step != 0:
            count += 1
            continue
        if not ret:
            print('No video signal, operation has broken')
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User\'s interrupt, operation has broken')
            break
        if(success>=target_num):
            print('Got enough training data, operation has finished')
            break

        img=frame

        img_path = str(success+count_start) + '.jpg'

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64,64))

        for face in faces:
            try:
                face_x, face_y, face_w, face_h = face
                img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
                if color_channels==1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(output_path, img_path), img)
                print(os.path.join(output_path, img_path))
                success += 1
                print('Succeeded: ' + str(success))
            except:
                continue
        count += 1

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print('====================/Getting faces from video source/====================')
    print('\n')
else:
    #process static image
    print('====================/Getting faces from static images/====================')
    for person_path in os.listdir(img_source):
        # output path
        output_path = os.path.join('training_data/images/',person_path)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        print('Getting training images for id: ' + person_path)
        count=count_start
        success=0
        for img_path in os.listdir(os.path.join(img_source, person_path)):
            if success >= target_num:
                print('Got enough training data, operation has finished')
                break

            # use cv2.imread to read and convert to gray color.
            img = cv2.imread(img_source + '/' + person_path + '/' + img_path)
            faces = faceCascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(64,64))

            img_path = str(count_start+success) + '.jpg'

            for face in faces:
                try:
                    face_x, face_y, face_w, face_h = face
                    img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
                    if color_channels==1:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(output_path, img_path), img)
                    success = success+1
                    print('Succeeded: ' + str(success))
                except:
                    continue
            count+=1;
    print('====================/Getting faces from static images/====================')
    print('\n\n')