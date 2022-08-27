from collections import defaultdict
import cv2
from sklearn.cluster import DBSCAN
from ..dbyaml import DBYaml
import os.path


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


def resize_img(image, scale_percent: int = 500):
    width, height = image.shape
    if height >= scale_percent or width >= scale_percent:
        resize = image
    else:
        height_scale_percent = int(round(scale_percent / height))
        width_scale_percent = int(round(scale_percent / width))
        if width_scale_percent >= height_scale_percent:
            dsize = [(height * height_scale_percent), (width * height_scale_percent)]

            resize = cv2.resize(image, dsize, cv2.INTER_CUBIC)
        else:
            dsize = [(height * width_scale_percent), (width * width_scale_percent)]
            resize = cv2.resize(image, dsize, cv2.INTER_CUBIC)
    return resize


def preprocessing_img(img, debag: bool = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize_gray = resize_img(gray)
    median = cv2.medianBlur(resize_gray, 5)
    _, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black, white = cv2.calcHist([thresh], [0], None, [2], [0, 256])
    if white > black:
        if debag is True:
            cv2.imshow("gray", gray)
            cv2.imshow("thresh", thresh)
            cv2.waitKey()
        img = thresh
    else:
        _, thresh_trunc = cv2.threshold(median, 127, 255, cv2.THRESH_TRUNC)
        if debag is True:
            cv2.imshow("thresh", thresh_trunc)
            cv2.waitKey()
        img = thresh_trunc

    return img


def easy_ocr_get_text(img_list: list, reader, debag_mode: bool = False):
    text_list = []
    for img in img_list:
        img = preprocessing_img(img, debag_mode)
        bounds = reader.readtext(img)
        text_in_box = []
        for text in bounds:
            text_in_box.append(' ' + text[1])

        text_list.append("".join(text_in_box))
    return text_list


def get_text(path_img, reader):
    path_yaml = DBYaml.from_image_path_to_yaml_path(path_img)
    circumcised = cut_text_boxes(path_img=path_img)

    if circumcised is not False:
        text = easy_ocr_get_text(img_list=circumcised, reader=reader)
        DBYaml.damp.text(text_list=text,
                         path=path_yaml)
    else:
        print("регионы не вырезаны ")


def test_box_from_net(reader, img_path: str, conf: dict):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    box_list = reader.detect(thresh, **conf)
    box_list = box_list[0][0]
    return box_list


def clustered_text_box(box_list, conf: dict):
    clusters_boxes = defaultdict(list)
    clustered = DBSCAN(**conf).fit_predict(box_list)
    clustered = clustered.tolist()
    for step in range(len(box_list)):
        _class = clustered[step]

        x0, x1, y0, y1 = box_list[step]
        value = [int(x0), int(y0), int(x1), int(y1)]
        clusters_boxes[_class].append(value)
    clusters_boxes = dict(clusters_boxes)
    return clusters_boxes


def union_boxs_region(clusters_boxes, single_regions: bool = False):
    depleted_regions = list()
    for key in clusters_boxes:

        if key == -1:
            if single_regions is True:
                group = clusters_boxes.get(key)
                depleted_regions.append(group)
                continue
            continue

        group = clusters_boxes.get(key)

        x, y = [], []
        for (x0, y0, x1, y1) in group:
            x.extend([x0, x1])
            y.extend([y0, y1])

        x_max, y_max, = map(lambda k: int(max(k)), [x, y])
        x_min, y_min, = map(lambda k: int(min(k)), [x, y])
        depleted_regions.append([x_min, y_min, x_max, y_max])
    return depleted_regions


def get_text_box_easyocr(img_path: str, reader, conf_craftNet: dict, conf_DBSCAN: dict, single_regions):
    box_list = test_box_from_net(reader=reader, img_path=img_path, conf=conf_craftNet)
    clusters_boxes = clustered_text_box(box_list=box_list, conf=conf_DBSCAN)
    depleted_regions = union_boxs_region(clusters_boxes)

    if clusters_boxes is not None and len(depleted_regions) != 0:
        DBYaml.damp.word_box(boxes=clusters_boxes,
                             path=DBYaml.from_image_path_to_yaml_path(
                                 img_path))

    if depleted_regions is not None and len(depleted_regions) != 0:
        DBYaml.damp.text_box(boxes=depleted_regions,
                             path=DBYaml.from_image_path_to_yaml_path(
                                 img_path))
