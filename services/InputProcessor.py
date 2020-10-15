import abc
import services.gesture.GestureValidatorService as gesture
import services.people.PeopleFilterService as people
import services.person.PersonValidatorService as person
import numpy as np
         

class InputProcessor:

    

    def __init__(self):
        self.gestureValidatorService=gesture.GestureValidationService()
        self.peopleFilterService=people.PeopleFilterService()
        self.personValidatorService=person.PersonValidatorService()
        
    
    def processStream(self,image,users,commands)->dict:
        people=self.peopleFilterService.get_people(image)
        validated_people=self.personValidatorService.is_user(people,users)
        return self.gestureValidatorService.calsifie_gesture(validated_people,commands)
        


