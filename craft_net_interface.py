import os
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.cluster import DBSCAN
from torch.autograd import Variable

from model_craft import (craft_utils, imgproc)
from model_craft.craft import CRAFT


@dataclass()
class Config:
    text_threshold: float
    link_threshold: float
    low_text: float
    mag_ratio: float
    canvas_size: int
    cuda: bool
    poly: bool


@dataclass()
class BoundingBox:
    boxes_from_net = None
    clusters_boxes = defaultdict(list)
    depleted_regions = list()
    single_regions = list()


class CraftInterface:
    """Класс CraftInterface используется для взаимодействия с обученными
    моделями нейронной сети Craft модель

    Основное применение - нахождения местоположения текста на изображениях.


    Attributes
    ----------
    bounding_boxes
        структура хранящая  координаты регионов текста в различных вариациях
        обработки
    net
        модель нейронной сети
    config
        конфигурация нейронной сети


     Methods
    -------
    edit_config(text_threshold: float = None,
                link_threshold: float = None,
                low_text: float = None,
                mag_ratio: float = None,
                canvas_size: int = None,
                cuda: bool = None,
                poly: bool = None):
        Изменение конфигурации нейронной сети
    net_initialization(trained_model_path: str)
        Инициализирует нейронную сеть, загружает обученные веса модели
    get_boxes(image_path: str, eps: int = 100, min_samples: int = 3)
        Получение и обработка координат регионов текста с нейронной сети,
        возвращает структуру bounding_boxes """

    def __init__(self):
        self.bounding_boxes = BoundingBox()
        self.net = None
        self.config = Config(text_threshold=0.7,
                             link_threshold=0.3,
                             low_text=0.4,
                             mag_ratio=1.5,
                             canvas_size=1280,
                             cuda=False,
                             poly=False)

    def edit_config(self, text_threshold: float = None,
                    link_threshold: float = None,
                    low_text: float = None,
                    mag_ratio: float = None,
                    canvas_size: int = None,
                    cuda: bool = None,
                    poly: bool = None):
        if text_threshold is not None:
            if type(text_threshold) != float:
                raise ValueError
            self.config.text_threshold = text_threshold

        if link_threshold is not None:
            if type(link_threshold) != float:
                raise ValueError
            self.config.link_threshold = link_threshold

        if low_text is not None:
            if type(low_text) != float:
                raise ValueError
            self.config.low_text = low_text

        if mag_ratio is not None:
            if type(mag_ratio) != float:
                raise ValueError
        self.config.mag_ratio = mag_ratio

        if canvas_size is not None:
            if type(canvas_size) != int:
                raise ValueError
        self.config.canvas_size = canvas_size

        if cuda is not None:
            if type(cuda) != bool:
                raise ValueError
        self.config.cuda = cuda

        if poly is not None:
            if type(poly) != bool:
                raise ValueError
        self.config.poly = poly

    def net_initialization(self, trained_model_path: str):
        self.net = CRAFT()
        if self.config.cuda:
            self.net.load_state_dict(
                    imgproc.copy_state_dict(torch.load(trained_model_path)))
        else:
            self.net.load_state_dict(imgproc.copy_state_dict(
                    torch.load(trained_model_path, map_location='cpu')))

        if self.config.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

    def _load_from_net(self, image_path: str, refine_net=None):
        image = imgproc.loadImage(image_path)
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
                image, self.config.canvas_size,
                interpolation=cv2.INTER_LINEAR,
                mag_ratio=self.config.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.config.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link,
                                               self.config.text_threshold,
                                               self.config.link_threshold,
                                               self.config.low_text,
                                               self.config.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        self.bounding_boxes.boxes_from_net = boxes.astype(int)

        return boxes, polys, ret_score_text

    def _clustering_box(self, eps: int, min_samples: int, ):
        box = self.bounding_boxes.boxes_from_net[:, 0]
        clustered = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(box)

        for step in range(len(self.bounding_boxes.boxes_from_net)):
            _class = clustered[step]
            value = self.bounding_boxes.boxes_from_net[step]
            self.bounding_boxes.clusters_boxes[_class].append(value)

    def _union_boxs(self):
        for key in self.bounding_boxes.clusters_boxes:
            group = self.bounding_boxes.clusters_boxes.get(key)

            if key == -1:
                self.bounding_boxes.single_regions.append(np.array(group))
                continue

            x_min, x_max, y_min, y_max = None, None, None, None
            count = True
            for subgroup in group:
                for (x, y) in subgroup:
                    if count is True:
                        x_min, x_max = x, x
                        y_min, y_max = y, y
                        count = False
                        continue

                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

            self.bounding_boxes.depleted_regions.append(
                    np.array([x_min, y_min, x_max, y_max]))

    def get_boxes(self, image_path: str, eps: int = 100, min_samples: int = 3):
        self._load_from_net(image_path)
        self._clustering_box(eps, min_samples)
        self._union_boxs()
        return self.bounding_boxes


# def draw_box_polys(path: str):
#     img = cv2.imread(path)
#     for pt in polys:
#         pt = pt.astype(int)
#
#         cv2.polylines(img, [pt], True, (0, 255, 255))
#     cv2.imshow('1', img)
#     cv2.imwrite(os.path.join('result', 'text_location.jpg'), img)
#     cv2.waitKey()


def draw_box(path: str, boxes, name: str = 'result'):
    img = cv2.imread(path)

    for (startX, startY, endX, endY) in boxes:
        # draw the bounding box on the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow('1', img)
    cv2.imwrite(os.path.join('result', name + '.jpg'), img)
    cv2.waitKey()


if __name__ == '__main__':
    image_path = 'B:/Data/git/CRAFT-pytorch/exampl/en_1.jpg'
    model_path = 'B:/Data/git/ocr-manga-translator/net/craft_ic15_20k.pth'
    net = CraftInterface()
    net.net_initialization(trained_model_path=model_path)
    net.get_boxes(image_path=image_path, eps=80)
    # draw_box(path=image_path,
    #          boxes=net.bounding_boxes.single_regions,
    #          name='1')
    draw_box(path=image_path,
             boxes=net.bounding_boxes.depleted_regions,
             name='2')
