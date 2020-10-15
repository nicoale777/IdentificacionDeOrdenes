
from event.ObserverInterface import *
from event.Event import *



class EventManager:
    def __init__(self):
        self.subscribers=[]
    
    def subscribe(self, subscriber:ObserverInterface):
        self.subscribers.append(subscriber)

    def unSubscribe(self, subscriber:ObserverInterface):
        self.subscribers.remove(subscriber)
    
    def notify(self, event:Event):
        for subscriber in self.subscribers:
            subscriber.update(event)