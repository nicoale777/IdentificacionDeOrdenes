import cv2
import numpy as np
from PIL import Image
from numpy import asarray


class PoeplFilter():

    def __init__(self):
        self.wth=416
        self.confidenceThreshold=0.5
        self.nmsThreshold=0.3
        modelConf='C:/Users/nicoa/OneDrive/Documentos/ReconocimientoDeOrdenes/yolo3/yolov3.cfg'
        modelWeights='C:/Users/nicoa/OneDrive/Documentos/ReconocimientoDeOrdenes/yolo3/yolov3.weights'

        self.net= cv2.dnn.readNetFromDarknet(modelConf,modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def get_people(self,img):
        blob=cv2.dnn.blobFromImage(img,1/255,(self.wth,self.wth),[0,0,0],1,crop=False)
        self.net.setInput(blob)
        layerNames=self.net.getLayerNames()
        outputNames= [layerNames[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        outputs=self.net.forward(outputNames)
        resu=self.findPeople(outputs,img)
        return  resu

    def findPeople(self,outputs,img):
        ht, wt, ct= img.shape
        bbox=[]
        classIds=[]
        confs=[]

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId=np.argmax(scores)
                confidence=scores[classId]

                if confidence>self.confidenceThreshold and classId==0:
                    w,h=int(det[2]*wt), int(det[3]*ht)
                    x,y=int((det[0]*wt)-w/2),int((det[1]*ht)-h/2)#centro
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        
        #print(len(bbox))
        indices=cv2.dnn.NMSBoxes(bbox,confs,self.confidenceThreshold,self.nmsThreshold)

        imgr=[]

        for i in indices:
            
            i=i[0]
            box=bbox[i]
            x,y,w,h=box[0],box[1],box[2],box[3]
            x1, y1 = abs(x), abs(y)
            x2, y2 = x1 + w, y1 + h 
            imgT = img[y1:y2, x1:x2]
            try:
                image=Image.fromarray(imgT)
                image=image.resize(size=(400,600))
                imgr.append(asarray(image))
            except:
                pass
            
        return imgr

# cap = cv2.VideoCapture(0)

# poeplFilter=PoeplFilter()   

# while True:
#     sucess, img=cap.read()

#     cuts=poeplFilter.get_people(img)

#     if len(cuts)>0:
#         cv2.imshow('Image', cuts[0])
#     else:
#         cv2.imshow('Image', img)

#     cv2.waitKey(1)