
class AspectTerm(object):
    def __init__(self, term, polarity, from_position, to_position):
        self.term = term
        self.polarity = polarity
        self.from_position = from_position
        self.to_position = to_position


class AspectCategory(object):
    def __init__(self, category_aspect, polarity):
        self.category_aspect = category_aspect
        self.polarity = polarity


class AspectExample(object):
    def __init__(self, id, text, aspect_terms, aspect_categories):
        self.id = id
        self.text = text
        self.aspect_terms = aspect_terms
        self.aspect_categories = aspect_categories

        present_categories = {}
        for category in aspect_categories:
            category_name = category.category_aspect
            if category_name in present_categories:
                raise Exception("The Aspect example already contains the category:" + str(category_name))
            present_categories[category_name] = category

        self.present_categories = present_categories

