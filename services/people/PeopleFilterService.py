from typing import List
import services.people.PeopleFilterImp as filter

class PeopleFilterService:

    def __init__(self):
        self.poepleFilter=filter.PeopleFilterImp()

    def get_people(self, image) -> List:
        return self.poepleFilter.detect_people(image)