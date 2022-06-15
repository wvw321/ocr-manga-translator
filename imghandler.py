import os.path
import time
from collections import defaultdict

import cv2
import easyocr
from sklearn.cluster import DBSCAN

from craft_net_interface import CraftInterface
from db_yaml import DBYaml
from config import IMAGE_DIRECTORY

from PIL import ImageDraw, Image


class ImageHandler:

    def __init__(self,
                 image_directory: str = None,
                 ):

        if isinstance(image_directory, str) is False:
            raise ValueError

        self.image_directory = image_directory
        self.img_names: list
        self.text_detector_model_craft_net = CraftInterface()

    @staticmethod
    def cut_text_boxes(path_yaml: str, path_img: str):
        if os.path.exists(path_yaml):
            text_box = DBYaml.load.text_box(path_yaml=path_yaml)
            img = cv2.imread(path_img)

            circumcised_img_text = []
            for key in text_box:
                x1 = text_box[key][0]
                x2 = text_box[key][2]
                y1 = text_box[key][1]
                y2 = text_box[key][3]
                crop = img[y1:y2, x1:x2]
                circumcised_img_text.append(crop)

            return circumcised_img_text
        return False

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

    def get_text_box(self, path_img: str = None):
        reader = easyocr.Reader(['en'])

        conf_DBSCAN = {'eps'        : 110,
                       'min_samples': 2}

        conf_craftNet = {"text_threshold": 0.6,
                         "link_threshold": 0.3,
                         "canvas_size"   : 1280,
                         "mag_ratio"     : 1.5,
                         "low_text"      : 0.4,
                         }

        def _get_text_box_easyocr(img_path: str):
            box_list = list()
            clusters_boxes = defaultdict(list)
            depleted_regions = list()

            def _test_box_from_net():
                nonlocal box_list
                box_list = reader.detect(img_path, **conf_craftNet)
                box_list = box_list[0][0]

            def _clustered_text_box():
                nonlocal clusters_boxes
                clustered = DBSCAN(**conf_DBSCAN).fit_predict(box_list)

                for step in range(len(box_list)):
                    _class = clustered[step]
                    value = box_list[step]
                    clusters_boxes[_class].append(value)

            def _union_boxs_region():
                nonlocal depleted_regions
                for key in clusters_boxes:

                    if key == -1:
                        continue

                    group = clusters_boxes.get(key)

                    x = []
                    y = []
                    for (x0, x1, y0, y1) in group:
                        x.append(x0)
                        x.append(x1)
                        y.append(y0)
                        y.append(y1)

                    x_max = int(max(x))
                    x_min = int(min(x))
                    y_max = int(max(y))
                    y_min = int(min(y))

                    depleted_regions.append([x_min, y_min, x_max, y_max])

            _test_box_from_net()
            _clustered_text_box()
            _union_boxs_region()
            if depleted_regions is not None and len(depleted_regions) != 0:
                DBYaml.damp.text_box(boxes=depleted_regions,
                                     path=DBYaml.from_image_path_to_yaml_path(
                                             img_path))

        if path_img is not None:
            _get_text_box_easyocr(img_path=path_img)
        else:
            self._links_loader()
            for img_name in self.img_names:
                _img_path = self.image_directory + "/" + img_name
                _get_text_box_easyocr(img_path=_img_path)

    def get_text(self, path_img: str = None, method: str = "easyocr"):

        def _preprocessing_img(img):

            cv2.imshow("1", img)
            cv2.waitKey()

            return img

        def easy_ocr_get_text(img_list: list):
            text_list = []
            for img in img_list:
                img = _preprocessing_img(img)

                bounds = reader.readtext(img)
                text_in_box = []
                for text in bounds:
                    text_in_box.append(' ' + text[1])

                text_list.append("".join(text_in_box))
            return text_list

        def _get_text():
            path_yaml = DBYaml.from_image_path_to_yaml_path(path_img)
            circumcised = ImageHandler.cut_text_boxes(
                    path_yaml=path_yaml,
                    path_img=path_img)

            if circumcised is not False:
                text = easy_ocr_get_text(circumcised)
                DBYaml.damp.text(text_list=text,
                                 path=path_yaml)
            else:
                print("регионы не вырезаны ")

        if method == "easyocr":
            reader = easyocr.Reader(['en'])
            if path_img is not None:
                _get_text()
            else:
                self._links_loader()
                for img_name in self.img_names:
                    path_img = self.image_directory + "/" + img_name
                    _get_text()

    def text_correct(self):
        pass

    def text_translator(self):
        pass

    def text_draw(self):
        pass

    def img_save(self):
        pass

    @staticmethod
    def draw_box(img_path, print_text: bool = False):
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        boxes = DBYaml.load.text_box(
                DBYaml.from_image_path_to_yaml_path(img_path))

        for key in boxes:
            bound = boxes.get(key)

            draw.rectangle(bound, outline=(0, 0, 0))

        return image.show()


if __name__ == '__main__':
    start_time = time.time()
    img_path = "example/4.jpg"
    # ImageHandler.draw_box(img_path)

    test = ImageHandler(image_directory=IMAGE_DIRECTORY)
    test.get_text_box()
    test.get_text()

    print("--- %s seconds ---" % (time.time() - start_time))
    # ImageHandler.draw_box(img_path)
