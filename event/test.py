import time

var=time.time()

current_milli_time = lambda: int(round(var * 1000))

print(var, current_milli_time())


""" class test(ObserverInterfase):
    def update (self, event:Event):
        print(event)

a=test()
a.update(Event.Event('asdf',23423))         """