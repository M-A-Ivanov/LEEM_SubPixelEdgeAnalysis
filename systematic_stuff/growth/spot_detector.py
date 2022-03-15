from matplotlib import pyplot as plt
from skimage.feature import blob_dog, blob_doh, blob_log

from image_processing import ImageProcessor
import numpy as np
import cv2
from skimage import io
from raw_reader import RawReader
import os
from random import randint


# import tensorflow_hub as tfhub


class SpotFinder:
    def __init__(self):
        self.image = None
        self.segmented = None
        self.spot_info = []

    @staticmethod
    def get_info_template():
        return {"coordinates": None,
                "size": None}

    def invert_image(self):
        self.image = cv2.bitwise_not(self.image)

    def segment(self):
        """ CV method to segment. If not effective, try U-Net?"""
        thresh_C = 6  # 6x6 : best: 6/13 mean or 5/19 gauss, but gauss is more noisy
        blocksize = 7  # 8x2: 6/11 gauss
        self.segmented = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, blocksize, thresh_C)
        fig, ax = plt.subplots(2)
        ax[0].imshow(self.image, cmap='gray')
        ax[1].imshow(self.segmented, cmap='gray')
        plt.show()

    def load_image(self, image, are_spots_black=True):
        self.image = image
        if not are_spots_black:
            self.invert_image()
        self.spot_info = []
        self.segment()

    def find_spots(self):
        """Working function for detection of blobsies."""

        _, _, stats, _ = cv2.connectedComponentsWithStats(self.segmented, connectivity=4)
        filtered_boxes = []
        for component in stats:
            """Here, we set limits for how big and how small blobs can be, by controlling the h and w of bounding boxes.
            """
            x = component[cv2.CC_STAT_LEFT]  # The leftmost (x) coordinate which is the inclusive start of the
            # bounding box in the horizontal direction.
            y = component[cv2.CC_STAT_TOP]  # The topmost (y) coordinate which is the inclusive start of the bounding
            # box in the vertical direction.
            w = component[cv2.CC_STAT_WIDTH]  # The horizontal size of the bounding box
            h = component[cv2.CC_STAT_HEIGHT]  # The vertical size of the bounding box
            pixels = component[cv2.CC_STAT_AREA]  # The total area (in pixels) of the connected component
            if pixels > ((2 / 3) * h * w) and abs(h - w) < (1 / 2) * (
                    h + w) and h < 100 and w < 100 and h > 3 and w > 3:
                filtered_boxes.append((x, y, w, h, pixels))
                collect_info = dict()
                collect_info["size"] = pixels
                collect_info["coordinates"] = np.array([x + w / 2, y - h / 2])
                self.spot_info.append(collect_info)
        return filtered_boxes

    def plot_found_spots(self, boxes):
        marked_image = self.image.copy()
        for x, y, w, h, _ in boxes:
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (255, 255, 255), 1)

        plt.imshow(marked_image, cmap="gray")
        plt.show()


class FeatureSpotFinder(SpotFinder):
    def segment(self):
        return

    def get_blobs(self):
        return None

    def find_spots(self):
        blobs = self.get_blobs()
        blobs = np.sqrt(2) * blobs
        filtered_boxes = []
        for blob in blobs:
            y, x, r = blob
            if r > 2:
                size = np.pi * r * r
                filtered_boxes.append((int(x - r / 2), int(y - r / 2), int(r), int(r), size))
                self.spot_info.append({"size": size,
                                       "coordinates": np.array([x, y])})

        return filtered_boxes


class DoGSpotFinder(FeatureSpotFinder):
    def get_blobs(self):
        return blob_dog(self.image, threshold=0.3)


class DoHSpotFinder(FeatureSpotFinder):
    def get_blobs(self):
        return blob_doh(self.image, max_sigma=30, threshold=0.3)


class LoGSpotFinder(FeatureSpotFinder):
    def get_blobs(self):
        return blob_log(self.image, max_sigma=30, threshold=1)


def try_spot_detection():
    path = r"F:\cardiff cloud\OneDrive - Cardiff University\Data\LEEM\11092019\experiment\recording\growth\2098521"
    file = r"cropped and clean small cut and aligned.tif"
    images = io.imread(os.path.join(path, file))
    images = images[randint(0, len(images) - 1)]
    processor = ImageProcessor()
    spotty = LoGSpotFinder()
    processor.load_image(images)
    processor.clahe_hist_equal()
    processor.denoise_tv()
    processor.to_uint8()
    # processor.plot_all()
    spotty.load_image(processor.result(), are_spots_black=True)
    boxes = spotty.find_spots()
    spotty.plot_found_spots(boxes)


if __name__ == '__main__':
    try_spot_detection()
