import face_recognition, os, cv2
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

info = """
    Students Attendance system using face recognition
    
    Are You want to add Students (y/n)
"""
PATH = "/datasets"
NewEntry = input(info + '\n>> ')

if NewEntry == 'y' :
    n = int(input('Enter Number of Students >> '))
    for i in range(1,n+1):    
        img = input(f'Enter the Name of Student {i} >> ')
        print('Make Sure Your Face has sufficient light\nPress [ENTER] to take photo')
        input()
        cam = cv2.VideoCapture(0)
        result, image = cam.read()

        if result:
            cv2.imshow(img, image)
            cv2.imwrite(os.path.join(PATH , img + '.png'), image)
            cv2.waitKey(0)
            cv2.destroyWindow(img)
              
            image = face_recognition.load_image_file(PATH + '/' + img + '.png')
            encoding = face_recognition.face_encodings(image)
            if not encoding:
                print("No image detected. Please! try again")
                continue
            print(f'Student {i}.{img} instance stored to the Dataset')
        else:
            print("No image detected. Please! try again")

video_capture = cv2.VideoCapture(0)
print('\nStarting Camera to Detect Face :)\n Make Sure Your Face has sufficient light to Recognise..,')
totalStudents = 0
StudentsEncodings = []
StudentsName = []

for base, dirs, files in os.walk(PATH):
    for File in files:
        Name = File[:-4]
        image = face_recognition.load_image_file(PATH + '/' + File)
        encoding = face_recognition.face_encodings(image)
        
        if encoding:
            StudentsEncodings.append(encoding[0])
            StudentsName.append(Name)

face_locations = []
face_encodings = []
face_names = []
attendies = set()
students = set(StudentsName)
process_this_frame = True

while True :
    ret, frame = video_capture.read() 
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]

    if process_this_frame :
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

        face_names = []
        
        name = 'unknown'
        for face_encoding in face_encodings :
            matches = face_recognition.compare_faces(StudentsEncodings,face_encoding)
            face_distances = face_recognition.face_distance(StudentsEncodings,face_encoding)
            best_match_index = np.argmin(face_distances)
           
            if matches[best_match_index] :
                name = StudentsName[best_match_index]
                attendies.add(name)
            face_names.append(name)
    
    process_this_frame = not process_this_frame

    for (top,right,bottom,left), name in zip(face_locations,face_names) :
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),2)
        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(255,0,0),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)

    cv2.imshow('Fauceter Face Recognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        absenties = students - attendies
        # print(students,absenties)
        a = list(attendies)
        b = list(absenties)

        if len(a) != len(b):
            if len(a) > len(b):
                zero = [' ' for _ in range(len(a)-len(b))]
                b += zero
            else:
                zero = [' ' for _ in range(len(b)-len(a))]
                a += zero
                
        data = {'Attendies':a,
                'Absenties':b
                }
        
        data = pd.DataFrame(data)
        data.to_csv('Attendance.csv', index=False)
        print(data)
        
        break

video_capture.release()
cv2.destroyAllWindows()