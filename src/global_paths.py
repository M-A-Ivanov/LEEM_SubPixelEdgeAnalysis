import os
import pickle


def load_pickle(picklepath):
    with open(picklepath, "rb") as f:
        return pickle.load(f)


def save_pickle(picklepath, thingy):
    with open(picklepath, "wb") as f:
        pickle.dump(thingy, f)


EDGE = [
        'edge 1',
        # 'edge 2',
        # 'edge 3  excl',
        # 'edge 4',
        # 'edge 5',
        # 'edge 6'
        ]

# CLOUD_PATH = r"J:\11022021"
# CLOUD_PATH = r"F:\cardiff cloud\OneDrive - Cardiff University"  # burgas
CLOUD_PATH = r"C:\Users\c1545871\OneDrive - Cardiff University"  # uni computer
# CLOUD_PATH = r"C:\Users\User\OneDrive - Cardiff University"  # laptop

DANIDATA_PATH = CLOUD_PATH + r"\Data\OriginData"
# REGION = 'region 3, 15fps, try 12'
REGION = 'run 6'
# SRC_FOLDER = os.path.join(CLOUD_PATH, "Data", "LEEM", r"07032022", r"fluctuations")
# SRC_FOLDER = os.path.join(r"C:\Users\c1545871\OneDrive - Cardiff University\Data\LEEM\11122021\fluctuations")
SRC_FOLDER = os.path.join(r"E:\Cardiff LEEM\Raw_images\11032022")
#
# SRC_FOLDER = os.path.join(r"E:\Cardiff LEEM\Raw_images\11122021\fluctuations")
RESULTS_FOLDER = os.path.join(CLOUD_PATH, r"UNIVERSITY\PhD\coexistence paper\results")
TARGET_FOLDER = os.path.join(RESULTS_FOLDER, REGION)
