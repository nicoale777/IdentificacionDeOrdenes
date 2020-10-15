from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from numpy import array
from numpy import expand_dims
import cv2
from keras.models import load_model
import numpy as np
import users.person as person

class PersonValidatorImp():

    def __init__(self):
        self.model = load_model('C:/Users/nicoa/OneDrive/Documentos/ReconocimientoDeOrdenes/faceRecognition_facenet/facenet_keras.h5')
        self.detector = MTCNN()
        self.anterior=None
        self.color = (255, 0, 0) #BGR 0-255 
        self.stroke = 2
    
    def detect_user(self, image, people:dict)->person.Person:
        results = self.detector.detect_faces(image)
        user=None
        if len(results)>0:
            x1, y1, width, height = results[0]['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            face = image[y1:y2, x1:x2]
            imageT=Image.fromarray(face)
            imageT=imageT.resize(size=(160,160))
            face=asarray(imageT)
            embbeding=self.get_embbedding(self.model,face)
            cv2.imshow('frame',face)
            cv2.waitKey(1)

            positive=None
            distanceActual=100000000
            for clie in people:
                
                distance= np.mean(np.square(embbeding - people[clie].faceEmbbeding))
                
                print(distance)             
                if distance < 0.3 and distance < distanceActual:
                    positive=people[clie].id 
                    distanceActual=distance

            cv2.rectangle(image, (x1, y1), (x2, y2), self.color, self.stroke)
            
            if positive is not None:
                user=person.Person()
                user.id=positive
                user.faceEmbbeding=embbeding
                user.image=image
        
        return user

    def get_embbedding(self,model, face_pixels):
        
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        #if self.anterior is not None:
        #    print(tf.reduce_mean(tf.square(yhat[0] - self.anterior)))
        #self.anterior=yhat[0]
        
        return yhat[0]

