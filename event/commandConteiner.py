import event.command as comm
import event.Event as ev

class CommandConteiner:
    def __init__(self):
        self.__commandData={}
        tmp=comm.Command
        tmp.id=1
        tmp.commandName='release'
        tmp.commandEmbbeding=[]
        tmp.event=ev.Event()
        tmp.event.name="release"
        self.__commandData.update({1:tmp})
        

    def get_commands(self):
        return self.__commandData
    
    def addCommand(self,command):
        self.__commandData.update(command.id,command)
    
    def removeUser(self,id):
        try:
            self.__commandData.pop(id)
            return True
        except:
            return False
    
    