import abc

class PeopleFilterInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'detect_people') and 
                callable(subclass.detect_people
                ) or 
                NotImplemented)
    
    @abc.abstractmethod
    def detect_people(self, image)->list:
        raise NotImplementedError