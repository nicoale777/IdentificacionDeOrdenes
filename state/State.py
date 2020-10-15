import abc
import state.Context as ctx
import event.Event as ev
import cv2
import time

class StateInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'freeControl') and 
                callable(subclass.freeControl
                ) and 
                hasattr(subclass, 'takeControl') and 
                callable(subclass.takeControl
                )and 
                hasattr(subclass, 'suspend') and 
                callable(subclass.suspend
                )or 
                NotImplemented)
    
    @abc.abstractmethod
    def freeControl (self):
        raise NotImplementedError
            
    @abc.abstractmethod
    def takeControl (self):
        raise NotImplementedError
            
    @abc.abstractmethod
    def suspend (self):
        raise NotImplementedError

class Active(StateInterface):

    def __init__(self):
        self.iterate=False

    def freeControl (self):
        self.iterate=False
        ctx.Context.state=ctx.Context.idle
        ctx.Context.state.freeControl()

            
    def takeControl (self):
        ctx.Context.eventManager.notify(ev.Event('Control taken',time.time))
        i=0
        self.iterate=True
        while self.iterate:
            img = ctx.Context.camera.get_frame()
            people=ctx.Context.processor.processStream(img, ctx.Context.usersConteiner.get_users(),commands=[])
           # if(len(people)>0):
           #     for key in people:                
           #         cv2.imshow('Image', people[key].image)
           # else:
           #     cv2.imshow('Image', img)
            cv2.imshow('Image', img)
            cv2.waitKey(1)
            i+=1
            if i>50:
                self.freeControl()
    
    def suspend (self):
        self.iterate=False

class Idle(StateInterface):

    def __init__(self):
        self.iterate=False

    def freeControl (self):
        ctx.Context.eventManager.notify(ev.Event('Control released',time.time))
        i=0
        self.iterate=True
        while self.iterate :
            img = ctx.Context.camera.get_frame()
            people=ctx.Context.processor.processStream(img, ctx.Context.usersConteiner.get_users(),commands=[])
            # if(len(people)>0):
           #     for key in people:                
           #         cv2.imshow('Image', people[key].image)
           # else:
           #     cv2.imshow('Image', img)
            cv2.imshow('Image', img)
            

            cv2.waitKey(1)
            i+=1
            if i>50:
                
                self.takeControl()

            
            
    def takeControl (self):
        self.iterate=False
        ctx.Context.state=ctx.Context.active        
        ctx.Context.state.takeControl()
    
    def suspend (self):
        self.iterate=False