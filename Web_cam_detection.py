import face_recognition
import numpy as np 
import cv2
import os

path="webcam"
images=[]
classname=[]
mylist=os.listdir(path)
print(mylist)


for cl in mylist:
    currim=cv2.imread(f"{path}/{cl}")
    images.append(currim)
    classname.append(os.path.splitext(cl)[0])
print(classname)    



def findincoding(images):
    encodelist=[]
    for img in images:
        imgs=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(imgs)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown=findincoding(images)
print(len(encodelistknown))
cap=cv2.VideoCapture(0)
while True:
    succ,img=cap.read()
    imgss=cv2.resize(img,(0,0),None,0.25,0.25)
    imgss=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    loc=face_recognition.face_locations(imgss)
    encode_of_cam=face_recognition.face_encodings(imgss,loc)
    for encodes,facecode in zip(encode_of_cam,loc):
        mathch=face_recognition.compare_faces(encodelistknown,encodes)
        faceDis = face_recognition.face_distance(encodelistknown,encodes)
        #print(faceDis)
        mach=np.argmin(faceDis)  #its return 0,1,2
        if mathch[mach]:
            name= classname[mach].upper()
            print(name)
            y1,x2,y2,x1=facecode
            print(y1,x2,y2,x1)
            
            cv2.rectangle(img,(x1*4,y1*4),(x2*4,y2*4),(255,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow("webcam",img)        
    cv2.waitKey(1)


    


# imtr=face_recognition.load_image_file("train.jpg")
# imtr=cv2.cvtColor(imtr,cv2.COLOR_BGR2RGB)
# im_resize1=cv2.resize(imtr,(500,400))

# imtest=face_recognition.load_image_file("train2.jpg")
# imtest=cv2.cvtColor(imtest,cv2.COLOR_BGR2RGB)
# imt=cv2.resize(imtr,(500,400))