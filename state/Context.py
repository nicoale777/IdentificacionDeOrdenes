import abc
from event.EventManager import *
from state.Context import *
import state.State as stat
import sys
import util.cameraUtil as camUtil
import services.InputProcessor as process
import users.usersContainer as cont


class Context(object):
    __metaclass__ = abc.ABCMeta
 
    state = 1 #class attribute to be used as the singleton's attribute
    eventManager=EventManager()
    camera=None
    state=None
    idle=stat.Idle()
    active=stat.Active()
    processor=process.InputProcessor()
    usersConteiner=cont.UsersConteiner()

    @abc.abstractmethod
    def __init__(self):
        pass #this prevents instantiation!
 
    @classmethod
    def printSelf(cls):
        print(cls.state) #prints out the value of the singleton's state
    @classmethod
    def otro(cls):
        cls.state+=1
    @classmethod
    def set_camera(cls, camera:camUtil.CameraInterface):
        cls.camera=camera
        cls.state=cls.idle
        cls.state.freeControl()
    
    @classmethod
    def stop(cls):
        cls.idle.suspend()
        cls.active.suspend()

    
 


