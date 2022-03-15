# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:37:46 2020

@author: User
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import os.path, sys
from matplotlib import ticker
import matplotlib


class BigDaddy(object):
    def __init__(self, path, tiff, blocksize, thresh_const, pltsave=True):
        self.path = path
        self.tiff_path = path + tiff + '.tif'
        self.bs = blocksize
        self.C = thresh_const
        self.imagesci = self.tifftoImages(self.tiff_path)

        self.img = None
        self.binary_img = None
        if tiff == '6x6':
            self.inverse = True
        elif tiff == '8x2':
            self.inverse = False

        self.pltsave = pltsave
        self.stats_collector = []

    @staticmethod
    def tifftoImages(tiff_path):
        imgs = io.imread(tiff_path)
        return (imgs / (np.max(imgs) / 255)).astype('uint8')

    def dodala_patchsize(self, rnge_min, rnge_max):
        for i in range(rnge_min, rnge_max):
            self.threshold_img(i)
            self.analysis_patchsize(self.custom_detection(), i)

    def dodala_freq(self, rnge_min, rnge_max):
        for i in range(rnge_min, rnge_max):
            self.threshold_img(i)
            self.analysis_frequency(self.custom_detection(), i)

    def dodala_coverage(self, rnge_min, rnge_max):
        if not self.inverse:
            self.threshold_img(0)
            self.steps_img = self.binary_img
        else:
            self.steps_img = 0

        for i in range(rnge_min, rnge_max):
            self.threshold_img(i)
            self.analysis_coverage(i)

    def threshold_img(self, i):
        self.img = self.imagesci[i, 50:125, 123:]
        if self.inverse:
            self.binary_img = cv2.adaptiveThreshold(cv2.bitwise_not(self.img), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY_INV, self.bs, self.C)
        else:
            self.binary_img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY_INV, self.bs, self.C)

    def blob_detection(self, i):
        """Seems to not work for such small blobs (in terms of pixels), but 
        keeping the function here anyways, might be useful in future"""
        img, thrimg = self.threshold_img(i)
        # Setting the parameters:
        p = cv2.SimpleBlobDetector_Params()

        p.filterByColor = True
        p.blobColor = 255
        p.filterByArea = True
        p.minArea = 2
        p.maxArea = 5000

        p.filterByCircularity = False
        p.minCircularity = 0.1

        # Detecting blobs
        detector = cv2.SimpleBlobDetector_create(p)
        keypoints = detector.detect(thrimg)

        img_with_circles = cv2.drawKeypoints(thrimg, keypoints,
                                             np.array([]), (0, 0, 255),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.namedWindow('blobimg', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blobimg', 1500, 150)
        cv2.imshow("blobimg", img_with_circles)

    def custom_detection(self):
        """Working function for detection of blobsies"""
        _, _, boxes, _ = cv2.connectedComponentsWithStats(self.binary_img, connectivity=4)
        filtered_boxes = []
        for x, y, w, h, pixels in boxes:
            if pixels > ((1 / 3) * h * w) and abs(h - w) < (1 / 3) * (h + w) and h < 50 and w < 50 and h > 2 and w > 2:
                filtered_boxes.append((x, y, w, h, pixels))

        return filtered_boxes

    def analysis_coverage(self, frame):
        x_dim = len(self.img[50, :])
        spacing = int(x_dim / 15)
        x = np.arange(spacing / 2, x_dim + (spacing / 2), spacing)
        coverage = np.zeros(len(x))
        for i in range(len(x)):
            coverage[i] = cv2.countNonZero(self.binary_img[:, i * spacing:(i + 1) * spacing]) / self.binary_img[:,
                                                                                                i * spacing:(
                                                                                                                        i + 1) * spacing].size
        self.plotScatter(self.img, [x, coverage], 0, frame)

    def analysis_frequency(self, filtered_boxes, i):
        locations = np.array([arr[0] for arr in filtered_boxes])
        largeness = np.array([arr[4] for arr in filtered_boxes])
        lala_stats = np.zeros((12, 2))
        assert (len(largeness) > 0)
        for i, large in enumerate(largeness):
            j = large // 10
            lala_stats[j, 0] += locations[i]
            lala_stats[j, 1] += 1
        x = np.arange(0, 120, 10)
        y = lala_stats[:, 0] / lala_stats[:, 1]
        # self.stats_collector.append(lala_stats)
        self.plotScatter(self.img, [x, y], 0, i)

    def analysis_patchsize(self, filtered_boxes, i):
        locations = np.array([arr[0] for arr in filtered_boxes])
        largeness = np.array([arr[4] for arr in filtered_boxes])

        segment = 0
        segment_width = 60

        if len(locations) == 0:
            return
        largenesses = []

        while (segment - segment_width) <= max(locations):
            if np.average(self.img[:, segment:(segment + segment_width)]) > 60:
                largenesses.append(np.mean(largeness[np.where((segment < locations) & (locations < (
                        segment + segment_width)))]))
            elif largenesses:
                break
            segment += segment_width
        largenesses = np.array(largenesses)
        bullshit = np.linspace(0, segment_width * len(largenesses), len(largenesses))

        self.plotScatter(self.img, [bullshit, largenesses], segment, i)
        img_with_rect = self.img.copy()

        for x, y, w, h, _ in filtered_boxes:
            cv2.rectangle(img_with_rect, (x, y), (x + w, y + h), (255, 255, 255), 1)
        self.plotImg([self.img, self.binary_img, img_with_rect],
                     ["Original Image", "Thresholded Binary Image", "Found spots in the image"], i)

    def plotScatter(self, original_img, scatter_points, max_segment, frame):
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [7, 1]})
        plt.subplots_adjust(hspace=0)
        axs[0].scatter(scatter_points[0], scatter_points[1],
                       marker='X', s=40, color='r')
        axs[0].plot(scatter_points[0], scatter_points[1])
        axs[0].axvline(max_segment, ls='--', color='r')
        axs[0].grid()
        axs[0].title.set_text("Mean area of spots in trail - frame %d" % frame)
        axs[0].set_ylabel("Pixels")
        axs[0].set_xlim(0, 825)
        axs[0].set_ylim(0, 1)
        axs[1].imshow(original_img, cmap='gray', aspect='auto')
        if self.pltsave:
            fig.savefig(self.path + "figs//figure_%.d" % frame)
            plt.close()
        else:
            plt.show()

    def plotImg(self, list_of_images, titles, frame):
        fig, axs = plt.subplots(len(list_of_images), figsize=(15, 10))
        if (len(list_of_images) == 1):
            axs.imshow(list_of_images[0], cmap='gray')
            axs.title.set_text(titles[0])
            axs.xaxis.set_major_locator(ticker.MaxNLocator(40))
        else:
            for i, img in enumerate(list_of_images):
                if len(img.shape) == 2:
                    axs[i].imshow(img, cmap='gray')
                    axs[i].title.set_text(titles[i] + '  - Frame %.d' % frame)
                if i != (len(list_of_images) - 1):
                    axs[i].xaxis.set_visible(False)
        if self.pltsave:
            plt.savefig(self.path + "images//img_%.d" % frame)
            plt.close()
        else:
            plt.show()

    def saveallimg(self):
        for i in range(0, 161):
            self.plotImg([self.img], ["%.d" % i])

            plt.savefig("%.d" % i)
            plt.close()


if __name__ == "__main__":
    plt.close('all')
    path = "C:\\Users\\User\\OneDrive - Cardiff University\\Data\\LEEM\\yuran_experiment\\"
    # for poopy_thr in range(4, 10):
    #     for self.bs in [3, 5, 7, 9, 11, 13]:
    tiff = '6x6'
    thresh_C = 6  # 6x6 : best: 6/13 mean or 5/19 gauss, but gauss is more noisy
    blocksize = 13  # 8x2: 6/11 gauss
    ob = BigDaddy(path, tiff, blocksize, thresh_C, pltsave=True)
    ob.dodala_coverage(0, 160)
