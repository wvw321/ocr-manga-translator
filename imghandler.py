class ImageHandler:

    def __init__(self, links: list = None):
        if type(links) != list:
            raise ValueError
        self.links = links

    def img_loader(self):
        pass

    def text_detector(self):
        pass

    def text_reader(self):
        pass

    def text_correct(self):
        pass

    def text_translator(self):
        pass

    def text_draw(self):
        pass

    def img_save(self):
        pass
