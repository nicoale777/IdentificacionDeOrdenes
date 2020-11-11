import numpy as np
import os
import tensorflow as tf

import random
from sklearn.utils import shuffle
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
import time
dir=[x[0] for x in os.walk('C:/Users/nicoa/OneDrive/Documentos/entrenamiento/')]
muestras={}
pose=0
X=[]
y=[]
count=0
for folder in dir:
    try:
        
# print(my_data.__class__)
#         print(my_data)
#         print(my_data.shape)
        my_data = np.genfromtxt(folder+'/salida.txt', delimiter='|')
        m,n=my_data.shape

        if n==14 :
            my_data = np.delete(my_data, 13, axis=1)
        my_data = np.delete(my_data, 0, axis=1)
            # print(my_data)
            # print(my_data.shape)   
        
        for muestra in my_data:
            
            #print(pose,muestra)
            #muestra.insert(0,count)
            #muestra=np.insert(muestra,0,count)
            X.append(muestra)
            y.append(pose)
            count+=1

        # my_data = np.delete(my_data, 0, axis=1)
        # print(my_data)
        # print(my_data.shape)
        

        pose+=1
    except:
        pass
    print(folder)

#print(X)
#print(y)
print(len(X),len(y))

X, y = shuffle(X, y)

#print(X)
#print(y)
#print(len(X),len(y))
y=np.asarray(y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.90,test_size=0.10)


def create_batch(batch_size=256):
    x_anchors = np.zeros((batch_size, 12))
    x_positives = np.zeros((batch_size, 12))
    x_negatives = np.zeros((batch_size, 12))
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, len(X_train)- 1)
        x_anchor = X_train[random_index]
        y = y_train[random_index]
        
        indices_for_pos = np.where(y_train == y)
        indices_for_neg = np.where(y_train != y)
        
        #print(len(indices_for_pos[0]))
        


        x_positive = X_train[indices_for_pos[0][random.randint(0, len(indices_for_pos[0]) - 1)]]
        x_negative = X_train[indices_for_neg[0][random.randint(0, len(indices_for_neg[0]) - 1)]]
        
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
    print(x_anchors.__class__)
    print(x_anchors.shape)
    return [x_anchors, x_positives, x_negatives]


print(create_batch())



emb_size = 32

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(emb_size, activation='sigmoid')
])

embedding_model.summary()

input_anchor = tf.keras.layers.Input(shape=(12,))
input_positive = tf.keras.layers.Input(shape=(12,))
input_negative = tf.keras.layers.Input(shape=(12,))

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
net.summary()

alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)

def data_generator(batch_size=256):
    while True:
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y

from sklearn.decomposition import PCA

class PCAPlotter(tf.keras.callbacks.Callback):
    
    def __init__(self, plt, embedding_model, x_test, y_test):
        super(PCAPlotter, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.fig = plt.figure(figsize=(9, 4))
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        plt.ion()
        self.i=0
        self.losses = []
        self.pl=plt
    
    def plot(self, epoch=None, plot_loss=False):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        pca_out = PCA(n_components=2).fit_transform(x_test_embeddings)
        self.ax1.clear()

        self.ax1.scatter(pca_out[:, 0], pca_out[:, 1], c=self.y_test, cmap='seismic')
        if plot_loss:
            self.ax2.clear()
            self.ax2.plot(range(epoch), self.losses)
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Loss')
        self.fig.canvas.draw()
        self.pl.savefig('C:/Users/nicoa/OneDrive/Documentos/testplot'+str(self.i)+'.png')
        self.i+=1
    def on_train_begin(self, logs=None):
        self.losses = []
        self.fig.show()
        self.fig.canvas.draw()
        self.plot()
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.plot(epoch+1, plot_loss=True)
print(X_test[:100].__class__, y_test[:100].__class__)
print(np.array(X_test[:100]).shape, np.array(y_test[:100]).shape)


batch_size = 128
epochs = 45
steps_per_epoch = int(len(X_train)/batch_size)

net.compile(loss=triplet_loss, optimizer='adam')

_ = net.fit(
    data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=True,
    callbacks=[
        PCAPlotter(
            plt, embedding_model,
            np.array(X_test[:300]), np.array(y_test[:300])
        )]
)

print('whait')



embedding_model.save("C:/Users/nicoa/OneDrive/Documentos/entrenamiento/")
loaded_model = tf.keras.models.load_model("C:/Users/nicoa/OneDrive/Documentos/entrenamiento/")

batch=create_batch(20)


for i in range(20):
    print("---------")
    example = np.expand_dims(batch[0][i], axis=0)
    anchor = embedding_model.predict(example)[0]
    example = np.expand_dims(batch[1][i], axis=0)
    test = embedding_model.predict(example)[0]    
    distPositivo=tf.reduce_mean(tf.square(anchor - test))
    example = np.expand_dims(batch[2][i], axis=0)
    test = embedding_model.predict(example)[0]
    distNegativo=tf.reduce_mean(tf.square(anchor - test))
    print(distPositivo,distNegativo)
    if distPositivo>distNegativo :
        print('mal')

print("************************************************************")

for i in range(20):
    print("---------")
    example = np.expand_dims(batch[0][i], axis=0)
    anchor = loaded_model.predict(example)[0]
    example = np.expand_dims(batch[1][i], axis=0)
    test = loaded_model.predict(example)[0]    
    distPositivo=tf.reduce_mean(tf.square(anchor - test))
    example = np.expand_dims(batch[2][i], axis=0)
    test = loaded_model.predict(example)[0]
    distNegativo=tf.reduce_mean(tf.square(anchor - test))
    print(distPositivo,distNegativo)
    if distPositivo>distNegativo :
        print('mal')

#tf.saved_model.save(embedding_model, "C:/Users/nicoa/OneDrive/Documentos/entrenamiento/")


    
