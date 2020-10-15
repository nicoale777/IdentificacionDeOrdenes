import abc
class CameraInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_frame') and 
                callable(subclass.get_frame
                ) or 
                NotImplemented)
    
    @abc.abstractmethod
    def get_frame (self):
        raise NotImplementedError