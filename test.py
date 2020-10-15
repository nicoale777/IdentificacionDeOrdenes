import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from state.Context import *
from event.Event import *
from event.EventManager import *
from event.ObserverInterface import *
from pose.posenetPython.posenet import *
import cv2
import threading
import time
from util.cameraUtil import *



Context.printSelf()

class CameraWeb(CameraInterface):
    def __init__(self):
        self.camera=cv2.VideoCapture(0)

    def get_frame(self):
        sucess, img=self.camera.read()
        return img

class test(ObserverInterface):
    def update (self, event:Event):
        print('´me llego:',event.name)

def thread_function(name):
    a=test()
    Context.eventManager.subscribe(a)
    Context.eventManager.notify(Event('asdf',23423))
    Context.set_camera(CameraWeb())


#x = threading.Thread(target=thread_function, args=(1,))
#x.start()

#time.sleep(120)
#Context.stop()

a=test()
Context.eventManager.subscribe(a)
Context.eventManager.notify(Event('asdf',23423))
Context.set_camera(CameraWeb())
