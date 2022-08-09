import os.path
import time
from collections import defaultdict

import cv2
import easyocr
import numpy as np
from PIL import ImageDraw, Image
from sklearn.cluster import DBSCAN

from config import IMG_PATH
from db_yaml import DBYaml


class ImageHandler:
    """Класс ImageHandler используется для обработки страниц
        манги

        Основное применение - Перевод и наложение текста на изображение

        Note:
            На данный момент реализовано:
            1)Нахождение регеонов расположения текста
            2)Считывание текста
            3) очиска изображения от текста

        Attributes
        ----------

        Methods
        -------

        """

    def __init__(self,
                 image_directory: str = None,
                 ):
        if image_directory is not None:
            if isinstance(image_directory, str) is False:
                raise ValueError

        self.image_directory = image_directory
        self.img_names: list

    @staticmethod
    def cut_text_boxes(path_img: str, save_dir: str = None):

        path_yaml = DBYaml.from_image_path_to_yaml_path(path_img)
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

            if save_dir is not None:
                name_img = os.path.basename(path_img)
                name_img, img_format = os.path.splitext(name_img)
                count = 1
                for img in circumcised_img_text:
                    print(save_dir + '/' + name_img + '_' + str(count) + img_format)
                    cv2.imwrite(save_dir + '/' + name_img + '_' + str(count) + img_format, img)
                    count += 1

            return circumcised_img_text
        return False

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

    @staticmethod
    def draw_word_box(img_path):
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        boxes = DBYaml.load.word_box(
            DBYaml.from_image_path_to_yaml_path(img_path))

        for key in boxes:
            bound = boxes.get(key)
            for rect in bound:
                draw.rectangle(rect, outline=(0, 0, 0))

        return image.show()

    @staticmethod
    def create_mask(path_img: str, save_dir: str = None, metod: str = "box"):
        img = cv2.imread(path_img)
        h, w, _ = img.shape
        mask = np.zeros([h, w], np.uint8)
        color = (255)
        thickness = -1

        def _show():
            dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
            cv2.imshow("dst", dst)
            cv2.imshow("mask", mask)
            cv2.waitKey()

        if metod == "box":
            boxes = DBYaml.load.text_box(
                DBYaml.from_image_path_to_yaml_path(img_path))
            for key in boxes:
                x0, y0, x1, y1 = boxes.get(key)
                mask = cv2.rectangle(mask, [x0, y0], [x1, y1], color, thickness)
            _show()

        if metod == "word":
            boxes = DBYaml.load.word_box(
                DBYaml.from_image_path_to_yaml_path(img_path))
            for key in boxes:
                bound = boxes.get(key)
                for x0, y0, x1, y1 in bound:
                    mask = cv2.rectangle(mask, [x0, y0], [x1, y1], color, thickness)
            _show()



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

    def get_text_box(self, path_img: str = None, single_regions: bool = False):
        reader = easyocr.Reader(['en'])

        conf_DBSCAN = {'eps': 105,
                       'min_samples': 2}

        conf_craftNet = {"min_size": 10,  # текстовое поле фильтра меньше минимального значения в пикселях
                         "text_threshold": 0.4,  # порог достоверности текста
                         "low_text": 0.4,  # нижняя граница текста
                         "link_threshold": 0.3,  # порог достоверности ссылки
                         "canvas_size": 2000,  # максимальный размер изображения
                         "mag_ratio": 1.5,  # коэффициент увеличения изображения
                         "slope_ths": 0.1,  # максимальный наклон
                         "ycenter_ths": 0.5,  # максимальное смещение в направлении y
                         "height_ths": 0.5,  # Максимальная разница в высоте блока
                         "width_ths": 0.5,  # максимальное расстояние по горизонтали для объединения блоков
                         "add_margin": 0.05,  # расширить ограничивающие  на определенное значение
                         }

        def _get_text_box_easyocr(img_path: str):
            box_list = list()
            clusters_boxes = defaultdict(list)
            depleted_regions = list()

            def _test_box_from_net():
                nonlocal box_list

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # # cv2.imshow("gray", thresh)
                # # cv2.waitKey()
                box_list = reader.detect(thresh, **conf_craftNet)

                # box_list = reader.detect(img_path, **conf_craftNet)
                box_list = box_list[0][0]

            def _clustered_text_box():
                nonlocal clusters_boxes
                clustered = DBSCAN(**conf_DBSCAN).fit_predict(box_list)
                clustered = clustered.tolist()
                for step in range(len(box_list)):
                    _class = clustered[step]

                    x0, x1, y0, y1 = box_list[step]
                    value = [int(x0), int(y0), int(x1), int(y1)]
                    clusters_boxes[_class].append(value)
                clusters_boxes = dict(clusters_boxes)

            def _union_boxs_region():
                nonlocal depleted_regions
                for key in clusters_boxes:

                    if key == -1:
                        if single_regions is True:
                            group = clusters_boxes.get(key)
                            depleted_regions.append(group)
                            continue
                        continue

                    group = clusters_boxes.get(key)

                    x = []
                    y = []
                    for (x0, y0, x1, y1) in group:
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
            if clusters_boxes is not None and len(depleted_regions) != 0:
                DBYaml.damp.word_box(boxes=clusters_boxes,
                                     path=DBYaml.from_image_path_to_yaml_path(
                                         img_path))

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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            resize = cv2.resize(gray, [300, 300], cv2.INTER_CUBIC)
            ret, thresh = cv2.threshold(resize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            black, white = cv2.calcHist([thresh], [0], None, [2], [0, 256])
            if white > black:

                # ret, thresh1 = cv2.threshold(thresh1, 127, 255, cv2.THRESH_BINARY)
                # kernel = np.ones((3, 3), np.uint8)
                # closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=1)

                cv2.imshow("gray", gray)
                cv2.imshow("thresh", thresh)
                # cv2.imshow("closing", closing)
                cv2.waitKey()
                img = thresh
            else:
                median = cv2.medianBlur(resize, 5)
                ret, thresh = cv2.threshold(median, 50, 255, cv2.THRESH_BINARY)
                cv2.imshow("thresh", thresh)
                cv2.waitKey()
                img = thresh

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
            circumcised = ImageHandler.cut_text_boxes(path_img=path_img)

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


if __name__ == '__main__':
    start_time = time.time()
    # img_path = IMG_PATH
    img_path = "example/3.jpg"
    test = ImageHandler()
    test.get_text_box(img_path)
    # test.get_text(img_path)
    # ImageHandler.draw_box(img_path)
    ImageHandler.draw_word_box(img_path)
    # ImageHandler.cut_text_boxes(img_path, "example")
    ImageHandler.create_mask(img_path, metod='word')

print("--- %s seconds ---" % (time.time() - start_time))
