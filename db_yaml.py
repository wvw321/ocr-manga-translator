import yaml
from os import path as os
from numpy import ndarray


class DBYaml:
    def __init__(self, path_img: str = None, ):
        if path_img is not None:
            split_file_path, _ = os.splitext(path_img)

            self.path = split_file_path + ".yaml"
        else:
            self.path = None

    def text_box_damp(self, boxes: list):
        if boxes is not None and len(boxes) != 0:
            if self.path is not None:
                with open(self.path, "w") as fl:
                    key = 1
                    dict_text_box = {}
                    for box in boxes:
                        if isinstance(box, ndarray):
                            dict_text_box[str(key)] = box.tolist()
                            key += 1

                    yaml.dump({"boxes_group": dict_text_box}, fl,
                              default_flow_style=False)
            else:
                raise ValueError('path variable is empty')

    def text_box_load(self, file_path: str):
        with open(file_path) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            box = data["boxes_group"]
        return box


if __name__ == '__main__':
    a = DBYaml().text_box_load(file_path="example/4.yaml")
    print(a)
