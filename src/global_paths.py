import os
import pickle


def load_pickle(picklepath):
    with open(picklepath, "rb") as f:
        return pickle.load(f)


def save_pickle(picklepath, thingy):
    with open(picklepath, "wb") as f:
        pickle.dump(thingy, f)


class AxisNames:

    @staticmethod
    def fft():
        return {"x": r"$\frac{1}{q^{2}}$, $nm^{2}$",
                "y": r"$\langle |y_{q}|^{2}\rangle$, $nm^{2}$"}

    @staticmethod
    def distr():
        return {"x": r"Offsets $\Delta x$, nm",
                "y": r"Probability"}

EDGE = [
        'edge 1',
        # 'edge 2',
        # 'edge 3',
        # 'edge 4',
        # 'edge 5',
        # 'edge 6'
        ]

# CLOUD_PATH = r"J:\11022021"
# CLOUD_PATH = r"F:\cardiff cloud\OneDrive - Cardiff University"  # burgas
CLOUD_PATH = r"C:\Users\c1545871\OneDrive - Cardiff University"  # uni computer
# CLOUD_PATH = r"C:\Users\User\OneDrive - Cardiff University"  # laptop

DANIDATA_PATH = CLOUD_PATH + r"\Data\OriginData"
# REGION = '3microns, 0.1s, r1'
REGION = 'run 6'
# SRC_FOLDER = os.path.join(r"C:\Users\User\OneDrive - Cardiff University\Data\LEEM\27042021\fluctuations")

# SRC_FOLDER = os.path.join(CLOUD_PATH, "Data", "LEEM", r"07032022", r"fluctuations")
# SRC_FOLDER = os.path.join(r"E:\my stuff\fluctuations data\fluctuations")
SRC_FOLDER = os.path.join(r"E:\Cardiff LEEM\Raw_images\11032022")
# TARGET_FOLDER = os.path.join(r"E:\results", REGION)
RESULTS_FOLDER = os.path.join(CLOUD_PATH, r"UNIVERSITY\PhD\coexistence paper\results")
TARGET_FOLDER = os.path.join(RESULTS_FOLDER, REGION)
