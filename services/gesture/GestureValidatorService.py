from typing import Dict
from typing import List
import users.person as person
import services.gesture.GestureValidatorImp as gesture

class GestureValidationService:
    def __init__(self):
        self.validator=gesture.GestureValidationImp()

    def calsifie_gesture(self, people:dict, commands:list) -> dict:
        
        for key in people:
            print(key, '->', people[key])
            people[key].image, people[key].command=self.validator.detect_pose(people[key].image, commands)
            


        return people