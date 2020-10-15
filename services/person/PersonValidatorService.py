import services.person.PersonValidatorImp as validatores
import users.person as person
import numpy as np

class PersonValidatorService:
    def __init__(self):
        self.validator=validatores.PersonValidatorImp()

    def is_user(self, images, users:dict) ->dict:
        people={}
        for image in images:
            result=self.validator.detect_user(image,users)
            if result is not None:
                people.update({result.id:result})
        return people

        