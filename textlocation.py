import os
from collections import defaultdict

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from sklearn.cluster import DBSCAN


class BoundingBoxes:
    def __init__(self):
        self.raw_box = None
        self.clusters_boxes = defaultdict(list)
        self.depleted_regions = list()
        self.single_regions = list()


class TextLocation:
    def __init__(self, model_path: str):

        self.bounding_boxes = BoundingBoxes()
        self._image_to_net = None
        self.img = None
        self.model_path = model_path
        self.net = None

    def model_initialization(self):
        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        self.net = cv2.dnn.readNet(self.model_path)
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets

    def _load_img(self, img_path: str):
        self.__rW = None
        self.__rH = None
        # load the input image and grab the image dimensions
        image = cv2.imread(img_path)
        self.img = image.copy()

        (H, W) = image.shape[:2]
        # resized image width (should be multiple of 32)
        new_w = (W // 32) * 32
        # resized image height (should be multiple of 32)
        new_h = (H // 32) * 32
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        self.__rW = W / float(new_w)
        self.__rH = H / float(new_h)

        # resize the image and grab the new image dimensions
        self._image_to_net = cv2.resize(image, (new_w, new_h))

        (self.__H, self.__W) = self._image_to_net.shape[:2]

    def _loading_the_image_into_the_model(self):
        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layer_names = ["feature_fusion/Conv_7/Sigmoid",
                       "feature_fusion/concat_3"]
        blob = cv2.dnn.blobFromImage(image=self._image_to_net,
                                     scalefactor=1.0,
                                     size=(self.__W, self.__H),
                                     mean=(123.68, 116.78, 103.94),
                                     swapRB=True,
                                     crop=False,
                                     )
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(layer_names)
        return scores, geometry

    def _results_processing(self, scores, geometry, min_confidence: float):

        (numRows, numCols) = scores.shape[2:4]
        reacts = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scores_data[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
                end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                reacts.append((start_x, start_y, end_x, end_y))
                confidences.append(scores_data[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        self._boxes_from_net = non_max_suppression(np.array(reacts),
                                                   probs=confidences)

    def _return_to_home_coordinates(self):
        self.bounding_boxes.raw_box = np.empty_like(self._boxes_from_net)
        counter = 0

        for (start_x, start_y, end_x, end_y) in self._boxes_from_net:
            start_x = int(start_x * self.__rW)
            start_y = int(start_y * self.__rH)
            end_x = int(end_x * self.__rW)
            end_y = int(end_y * self.__rH)

            self.bounding_boxes.raw_box[counter] = (start_x, start_y,
                                                    end_x, end_y)
            counter += 1

    def _clustering_box(self, eps: int, min_samples: int, ):

        clustered = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
                self.bounding_boxes.raw_box)

        for x in range(len(self.bounding_boxes.raw_box)):
            _class = clustered[x]
            value = self.bounding_boxes.raw_box[x]
            self.bounding_boxes.clusters_boxes[_class].append(value)

    def _union_boxs(self):

        for key in self.bounding_boxes.clusters_boxes:
            group = self.bounding_boxes.clusters_boxes.get(key)
            if key == -1:
                for (startX, startY, endX, endY) in group:
                    self.bounding_boxes.single_regions.append(
                            np.array([startX, startY, endX, endY]))
                continue

            start_x_min = None
            start_y_min = None
            end_x_max = None
            end_y_max = None
            count = True
            for (startX, startY, endX, endY) in group:
                if count is True:
                    start_x_min = startX
                    start_y_min = startY
                    end_x_max = endX
                    end_y_max = endY
                    count = False

                if startX < start_x_min:
                    start_x_min = startX

                if startY < start_y_min:
                    start_y_min = startY

                if endX > end_x_max:
                    end_x_max = endX

                if endY > end_y_max:
                    end_y_max = endY

            self.bounding_boxes.depleted_regions.append(
                    np.array([start_x_min, start_y_min, end_x_max, end_y_max])
            )

    def get_coordinates(
            self,
            img_path: str,
            min_confidence: float = 0.7,
            eps: int = 100,
            min_samples: int = 3,
    ):
        self._load_img(img_path)
        (scores, geometry) = self._loading_the_image_into_the_model()
        self._results_processing(scores=scores,
                                 geometry=geometry,
                                 min_confidence=min_confidence
                                 )
        try:
            self._return_to_home_coordinates()
        except:
            print("_return_to_home_coordinates failed!")

        try:
            self._clustering_box(eps=eps,
                                 min_samples=min_samples)
        except:
            print("_clustering_box failed!")
        try:
            self._union_boxs()
        except:
            print("_union_boxs failed!")

        return self.bounding_boxes

    def drow_box(self, boxes):
        for (startX, startY, endX, endY) in boxes:
            # draw the bounding box on the image
            cv2.rectangle(self.img, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)


if __name__ == "__main__":
    a = TextLocation(model_path="frozen_east_text_detection.pb")
    a.model_initialization()
    a.get_coordinates(img_path="exampl/img_10.jpg",
                      min_confidence=0.1,
                      eps=100,
                      min_samples=2)
    a.drow_box(boxes=a.bounding_boxes.raw_box)

    cv2.imshow("Text Detection", a.img)
    path = "B:/Data/PycharmProjects/Manga-cv/rezult"
    cv2.imwrite(os.path.join(path, "textlocation.jpg"), a.img)
    cv2.waitKey(0)

    # color = {'-1': (0, 255, 0), '0': (0, 128, 128), '1': (255, 255, 0),
    #          '2': (255, 69, 0), '3': (199, 21, 133), '4': (255, 0, 0)}
    #
    # for key in a._boxes_sorted:
    #     group = a._boxes_sorted.get(key)
    #
    #     for (startX, startY, endX, endY) in group:
    #         cv2.rectangle(a.img, (startX, startY), (endX, endY),
    #                                    (color.get(str(key))), 2)
