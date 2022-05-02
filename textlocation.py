from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict


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
        newW = (W // 32) * 32  # resized image width (should be multiple of 32)
        newH = (H // 32) * 32  # resized image height (should be multiple of 32)
        # set the new width and height and then determine the ratio in change
        # for both the width and height
        self.__rW = W / float(newW)
        self.__rH = H / float(newH)

        # resize the image and grab the new image dimensions
        self._image_to_net = cv2.resize(image, (newW, newH))

        (self.__H, self.__W) = self._image_to_net.shape[:2]

    def _loading_the_image_into_the_model(self):
        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        blob = cv2.dnn.blobFromImage(self._image_to_net, 1.0, (self.__W, self.__H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(layerNames)
        return scores, geometry

    def _results_processing(self, scores, geometry, min_confidence: float):

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        self._boxes_from_net = non_max_suppression(np.array(rects), probs=confidences)

    def _return_to_home_coordinates(self):
        self.bounding_boxes.raw_box = np.empty_like(self._boxes_from_net)
        counter = 0
        for (startX, startY, endX, endY) in self._boxes_from_net:
            startX = int(startX * self.__rW)
            startY = int(startY * self.__rH)
            endX = int(endX * self.__rW)
            endY = int(endY * self.__rH)

            self.bounding_boxes.raw_box[counter] = (startX, startY, endX, endY)
            counter += 1

    def _clustering_box(self, eps: int, min_samples: int, ):
        clustered = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.bounding_boxes.raw_box)

        for x in range(len(self.bounding_boxes.raw_box)):
            _class = clustered[x]
            value = self.bounding_boxes.raw_box[x]
            self.bounding_boxes.clusters_boxes[_class].append(value)

    def _union_boxs(self):

        for key in self.bounding_boxes.clusters_boxes:
            group = self.bounding_boxes.clusters_boxes.get(key)
            if key == -1:
                for (startX, startY, endX, endY) in group:
                    self.bounding_boxes.single_regions.append(np.array([startX, startY, endX, endY]))
                continue

            startX_min = None
            startY_min = None
            endX_max = None
            endY_max = None
            count = True
            for (startX, startY, endX, endY) in group:
                if count is True:
                    startX_min = startX
                    startY_min = startY
                    endX_max = endX
                    endY_max = endY
                    count = False

                if startX < startX_min:
                    startX_min = startX

                if startY < startY_min:
                    startY_min = startY

                if endX > endX_max:
                    endX_max = endX

                if endY > endY_max:
                    endY_max = endY

            self.bounding_boxes.depleted_regions.append(np.array([startX_min, startY_min, endX_max, endY_max]))

    def get_coordinates(self, img_path: str, min_confidence: float = 0.7, eps: int = 100, min_samples: int = 3):
        self._load_img(img_path)
        (scores, geometry) = self._loading_the_image_into_the_model()
        self._results_processing(scores=scores,
                                 geometry=geometry,
                                 min_confidence=min_confidence)
        try:
            self._return_to_home_coordinates()
        except:
            print('_return_to_home_coordinates failed!')

        try:
            self._clustering_box(eps=eps,
                                 min_samples=min_samples)
        except:
            print('_clustering_box failed!')
        try:
            self._union_boxs()
        except:
            print('_union_boxs failed!')

        return self.bounding_boxes

    def drow_box(self, boxes):
        for (startX, startY, endX, endY) in boxes:
            # draw the bounding box on the image
            cv2.rectangle(self.img, (startX, startY), (endX, endY), (0, 255, 0), 2)


if __name__ == '__main__':
    a = TextLocation("frozen_east_text_detection.pb")
    a.model_initialization()
    a.get_coordinates(img_path='exampl/img_10.jpg',
                      min_confidence=0.1,
                      eps=100,
                      min_samples=2)
    a.drow_box(a.bounding_boxes.raw_box)

    # color = {'-1': (0, 255, 0), '0': (0, 128, 128), '1': (255, 255, 0),
    #          '2': (255, 69, 0), '3': (199, 21, 133), '4': (255, 0, 0)}
    #
    # for key in a._boxes_sorted:
    #     group = a._boxes_sorted.get(key)
    #
    #     for (startX, startY, endX, endY) in group:
    #         cv2.rectangle(a.img, (startX, startY), (endX, endY), (color.get(str(key))), 2)

    cv2.imshow("Text Detection", a.img)
    path = 'B:/Data/PycharmProjects/Manga-cv/rezult'
    cv2.imwrite(os.path.join(path, 'textlocation.jpg'), a.img)
    cv2.waitKey(0)
