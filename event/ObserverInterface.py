
import abc
from event.Event import *


class ObserverInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'update') and 
                callable(subclass.update
                ) or 
                NotImplemented)
    
    @abc.abstractmethod
    def update (self, event:Event):
        raise NotImplementedError