import yaml
import os.path
from numpy import ndarray


class DBYaml:
    @staticmethod
    def _checking_exceptions(str_value: str, list_value: list):

        if not isinstance(list_value, list):
            raise TypeError('list_value wrong type variable')

        if list_value is None or len(list_value) == 0:
            raise ValueError('list_value is empty')

        if not isinstance(str_value, str):
            raise TypeError('str_value wrong type variable')

        if str_value is None:
            raise ValueError('str_value variable is empty')

    @staticmethod
    def _damp_yaml(path: str, data, key: str = None):
        if key is None:
            with open(path, "w") as fl:
                yaml.dump(data, fl,
                          default_flow_style=False)
        else:
            with open(path, "w") as fl:
                yaml.dump({key: data}, fl,
                          default_flow_style=False)

    @staticmethod
    def _overwriting_yaml(key: str, dict_data: dict, path: str):

        _data: dict = DBYaml._load_yaml(path)
        dict_data = {key: dict_data}
        data = _data | dict_data
        DBYaml._damp_yaml(path=path,
                          data=data)

    @staticmethod
    def _load_yaml(path_yaml: str):
        with open(path_yaml) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data

    @staticmethod
    def from_image_path_to_yaml_path(path_img: str):
        split_file_path, _ = os.path.splitext(path_img)
        path = split_file_path + ".yaml"
        return path

    class damp:
        def __init__(self, path: str, data, kay: str = None):
            DBYaml._damp_yaml(path=path, key=kay, data=data)

        @staticmethod
        def _dict_create(dates):
            count = 1
            _dict_text_box = {}
            for data in dates:
                if isinstance(data, ndarray):
                    _dict_text_box[str(count)] = data.tolist()
                    count += 1
                    continue
                if isinstance(data, list) or isinstance(data, str):
                    _dict_text_box[str(count)] = data
                    count += 1
            return _dict_text_box

        @staticmethod
        def text_box(path: str, boxes: list):
            DBYaml._checking_exceptions(str_value=path, list_value=boxes)
            key = "boxes_group"
            dict_data = DBYaml.damp._dict_create(boxes)
            if os.path.exists(path):
                DBYaml._overwriting_yaml(key=key,
                                         dict_data=dict_data,
                                         path=path)
            else:
                DBYaml._damp_yaml(data=dict_data,
                                  path=path,
                                  key=key)

        @staticmethod
        def text(path: str, text_list: list):
            DBYaml._checking_exceptions(str_value=path, list_value=text_list)
            key = "text"
            dict_data = DBYaml.damp._dict_create(text_list)
            if os.path.exists(path):
                DBYaml._overwriting_yaml(path=path,
                                         key=key,
                                         dict_data=dict_data,
                                         )
            else:
                DBYaml._damp_yaml(path=path,
                                  key=key,
                                  data=dict_data)

    class load:
        def __init__(self, path_yaml: str):
            DBYaml._load_yaml(path_yaml=path_yaml)

        @staticmethod
        def _load(key: str, path_yaml: str):
            if os.path.exists(path_yaml):
                data = DBYaml._load_yaml(path_yaml)
                try:
                    data = data[key]
                    return data
                except KeyError:
                    print("variable '" + key + "' not found in yaml file")
                    return False

        @staticmethod
        def text_box(path_yaml: str):
            box = DBYaml.load._load("boxes_group", path_yaml)
            return box

        @staticmethod
        def text(path_yaml: str):
            text = DBYaml.load._load("text", path_yaml)
            return text


if __name__ == '__main__':
    a = DBYaml.load.text_box(path_yaml="example/4.yaml")
    b = DBYaml.load.text(path_yaml="example/4.yaml")
    print(a, b)
