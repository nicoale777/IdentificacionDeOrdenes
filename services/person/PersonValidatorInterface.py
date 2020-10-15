import abc
import users.person as person

class PersonValidatorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'detect_user') and 
                callable(subclass.detect_user
                ) or 
                NotImplemented)
    
    @abc.abstractmethod
    def detect_user(self, image, people:list)->person.Person:
        raise NotImplementedError