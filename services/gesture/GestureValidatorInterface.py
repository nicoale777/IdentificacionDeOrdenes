import abc
import users.person as person

class GestureValidationInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'detect_pose') and 
                callable(subclass.detect_pose
                ) or 
                NotImplemented)
    
    @abc.abstractmethod
    def detect_pose(self, image, commands:list):
        raise NotImplementedError