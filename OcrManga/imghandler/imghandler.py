import os.path

import easyocr
from PIL import ImageDraw, Image

from EasyLaMa import TextRemover
from ..dbyaml import DBYaml
from ..imghandler.functions import get_text_box_easyocr, get_text


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
                 debag_mode: bool = False
                 ):

        if image_directory is not None:
            if isinstance(image_directory, str) is False:
                raise ValueError
        self.debag_mode = debag_mode
        self.image_directory = image_directory
        self.img_names: list

    @staticmethod
    def draw_box(img_path):
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
        boxes = DBYaml.load.word_box(DBYaml.from_image_path_to_yaml_path(img_path))

        for key in boxes:
            bound = boxes.get(key)
            for rect in bound:
                draw.rectangle(rect, outline=(0, 0, 0))

        return image.show()

    @staticmethod
    def image_cleanup(path_img: str, mask_edge: int = 15, radius=1):
        image = Image.open(path_img)
        mask = Image.new(mode="L", size=image.size)
        draw = ImageDraw.Draw(mask)
        boxes = DBYaml.load.word_box(DBYaml.from_image_path_to_yaml_path(path_img))

        for key in boxes:
            bound = boxes.get(key)
            for box in bound:
                box = [coordinate - mask_edge for coordinate in box[:2]] + [coordinate + mask_edge for coordinate in
                                                                            box[2:]]
                draw.rounded_rectangle(xy=box, fill=255, outline=255, width=1, radius=radius)
        mask.show()
        Tr = TextRemover(device="cpu", easyocr=False).inpaint(image, mask)
        Tr.show()

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

        conf_DBSCAN = {'eps': 150,
                       'min_samples': 2}

        conf_craftNet = {"min_size": 10,  # текстовое поле фильтра меньше минимального значения в пикселях
                         "text_threshold": 0.6,  # порог достоверности текста
                         "low_text": 0.4,  # нижняя граница текста
                         "link_threshold": 0.3,  # порог достоверности ссылки
                         "canvas_size": 2500,  # максимальный размер изображения
                         "mag_ratio": 1.5,  # коэффициент увеличения изображения
                         "slope_ths": 0.1,  # максимальный наклон
                         "ycenter_ths": 0.5,  # максимальное смещение в направлении y
                         "height_ths": 0.5,  # Максимальная разница в высоте блока
                         "width_ths": 0.5,  # максимальное расстояние по горизонтали для объединения блоков
                         "add_margin": 0,  # расширить ограничивающие  на определенное значение
                         }

        if path_img is not None:
            get_text_box_easyocr(img_path=path_img,
                                 reader=reader,
                                 conf_craftNet=conf_craftNet,
                                 conf_DBSCAN=conf_DBSCAN,
                                 single_regions=single_regions)
        else:
            self._links_loader()
            for img_name in self.img_names:
                _img_path = self.image_directory + "/" + img_name
                get_text_box_easyocr(img_path=_img_path,
                                     reader=reader,
                                     conf_craftNet=conf_craftNet,
                                     conf_DBSCAN=conf_DBSCAN,
                                     single_regions=single_regions)

    def get_text(self, path_img: str = None, method: str = "easyocr"):

        if method == "easyocr":
            reader = easyocr.Reader(['en'])
            if path_img is not None:
                get_text(path_img=path_img, reader=reader)
            else:
                self._links_loader()
                for img_name in self.img_names:
                    path_img = self.image_directory + "/" + img_name
                    get_text(path_img=path_img, reader=reader)

    def text_correct(self):
        pass

    def text_translator(self):
        pass

    def text_draw(self):
        pass

    def img_save(self):
        pass
