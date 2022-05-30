import os.path
import time
from dataclasses import dataclass
from config import IMAGE_DIRECTORY, CRAFT_PATH
import easyocr
from cv2 import imread

from craft_net_interface import CraftInterface
from db_yaml import DBYaml


@dataclass
class Data:
    box = dict()
    text = dict()
    translate = dict()


class ImageHandler:

    def __init__(self,
                 image_directory: str = None,
                 craft_net_path: str = None,
                 ):

        if isinstance(image_directory, str) is False:
            raise ValueError
        self.image_directory = image_directory

        self.craft_net_path = os.path.normpath(craft_net_path)
        self.img_names: list
        self.text_detector_model_craft_net = CraftInterface()
        self.easyocr_model = None

    def _links_loader(self):
        #  является ли путь директорией.
        if os.path.isdir(self.image_directory) is True:
            links = os.listdir(self.image_directory)
            img_names = []
            for link in links:
                filename, file_extension = os.path.splitext(link)
                if file_extension == '.jpg' or file_extension == '.png':
                    img_names.append(str(link))
            self.img_names = img_names
        else:
            print("не верная директория ")

    def _text_detector_craft_net(self, image_path: str):

        if self.text_detector_model_craft_net.net is None:
            self.text_detector_model_craft_net.net_initialization(
                    trained_model_path=self.craft_net_path
            )
        self.text_detector_model_craft_net.get_boxes(image_path=image_path)
        boxes = self.text_detector_model_craft_net.bounding_boxes. \
            depleted_regions

        return boxes

    def _text_reader_easyocr(self, img):
        if self.easyocr_model is None:
            self.easyocr_model = easyocr.Reader(['en'])
        bounds = self.easyocr_model.readtext(img)
        return bounds

    def get_text_box(self):
        self._links_loader()
        for img_name in self.img_names:
            path_img = self.image_directory + "/" + img_name
            boxes = self._text_detector_craft_net(image_path=path_img)
            DBYaml(path_img=path_img).text_box_damp(boxes)

    def get_text(self, method: str = "easyocr"):

        for img_name in self.img_names:
            split_name, _ = os.path.splitext(img_name)
            path_yaml = self.image_directory + "/" + split_name + ".yaml"
            path_img = self.image_directory + "/" + img_name
            text_box = DBYaml().text_box_load(file_path=self.image_directory)
            img = imread(path_img)

    def text_correct(self):
        pass

    def text_translator(self):
        pass

    def text_draw(self):
        pass

    def img_save(self):
        pass


if __name__ == '__main__':
    start_time = time.time()
    test = ImageHandler(image_directory=IMAGE_DIRECTORY,
                        craft_net_path=CRAFT_PATH)
    test.get_text_box()
    print(test.img_names)
    print("--- %s seconds ---" % (time.time() - start_time))
