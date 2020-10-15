from keras.models import load_model
# load the model
model = load_model('C:/Users/nicoa/OneDrive/Documentos/ReconocimientoDeOrdenes/faceRecognition_facenet/facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)