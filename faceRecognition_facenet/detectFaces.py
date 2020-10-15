from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from numpy import expand_dims
import cv2
from keras.models import load_model
import tensorflow as tf
# load the model
model = load_model('C:/Users/nicoa/OneDrive/Documentos/ReconocimientoDeOrdenes/faceRecognition_facenet/facenet_keras.h5')

class Comparator:


    def __init__(self):
        self.anterior=None
        

    def get_embedding(self,model, face_pixels):
        
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        result='['
        for val in yhat[0]:
            result+=','+str(val)
        result+=']'
        #print(result)
        if self.anterior is not None:
            print(self.anterior.__class__.__name__, yhat[0].__class__.__name__)
                
            t=tf.reduce_mean(tf.square(yhat[0] - self.anterior))
            val=t.numpy()
            print(val)
        self.anterior=[-0.33390716,1.3819865,-0.07786481,0.70857847,0.24718358,0.7417411,0.97698104,-0.8840089,-0.8536886,-1.6427152,0.17553797,-1.2036891,-0.84762365,0.8139112,-0.3098194,-1.4106148,0.7365471,-1.4999126,-0.003845401,-0.6734084,-0.4854355,-2.0168424,0.73366505,1.0451871,1.0804613,2.079065,1.312016,0.5539365,-1.3197213,-2.8144832,0.4898471,1.2043251,-0.9075492,0.57338613,-0.13535471,0.741485,1.5864418,-1.5138168,1.5939019,0.7458855,-0.25728932,0.38229322,-0.39756003,-0.67741996,0.26351634,1.4532634,0.32795438,0.87015426,0.37662113,-0.5938536,0.36181045,1.0322195,1.0449563,0.60164535,0.19907184,3.8771787,0.88458866,-1.8499888,1.9905971,0.98225,0.027041145,0.48094094,1.1339432,0.19970831,-0.70137405,0.49351448,0.82966244,-0.1481972,0.5733744,1.1934911,-0.17521927,1.358254,0.15560865,0.8454792,-1.5476538,0.6864199,-0.6449822,0.06802265,-1.369061,-1.1555864,-0.15239954,0.43646178,-0.08202573,0.44963357,1.6666594,-0.1717355,1.2696829,-1.4751154,0.77597344,1.141186,-0.67582154,-1.0924326,0.23210987,-0.11443294,-0.6370521,1.647187,0.7431287,-0.6060557,0.51053494,1.2805771,-0.61904854,-0.8108455,0.82638705,-0.54971343,0.4846753,-0.44978845,-2.2947412,2.553987,-0.115006685,-1.2889227,-0.6236715,0.034572735,0.7468715,-0.13755155,0.29715896,2.3011444,-0.59883535,0.9664369,0.3402167,-0.5906422,1.0546356,-1.7566596,0.68090385,0.49795726,0.15574956,-0.6670555,1.8341055,-1.5963628]
        #yhat[0]
        
        return yhat[0]

detector = MTCNN()
cap = cv2.VideoCapture(0)
color = (255, 0, 0) #BGR 0-255 
stroke = 2
comparator=Comparator()
while(True):
    ret, frame = cap.read()
    # detect faces in the image
    results = detector.detect_faces(frame)#(pixels)
    if len(results)>0:
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = frame[y1:y2, x1:x2]
        image=Image.fromarray(face)
        image=image.resize(size=(160,160))
        face=asarray(image)
        # scale pixel values
        #face = face.astype('float32')
        # standardize pixel values across channels (global)
        #mean, std = face.mean(), face.std()
        #face = (face - mean) / std


        comparator.get_embedding(model,face)     
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, stroke)

    if face is not None:
        cv2.imshow('frame',face)
    else:
        cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    #image = Image.fromarray(face)
    #image = image.resize((160, 160))
    #face_array = asarray(image)

