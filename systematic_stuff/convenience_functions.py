from typing import List
import numpy as np

import cv2 as cv2
import json
import os
from datetime import datetime
from natsort import natsorted
import natsort
import tifffile
from skimage import io, img_as_uint
from tqdm import tqdm

from image_processing import ImageProcessor
from raw_reader import RawReader


def _preprocess_image(image, local_hist=None):
    process = ImageProcessor()
    process.load_image(image)
    if local_hist is None:
        process.normalize()
    elif local_hist:
        process.clahe_hist_equal()
    else:
        process.global_hist_equal()
    process.denoise_bilateral()
    return process.result(uint16=True)


def _cutting_fn(x_frac=0.5, y_frac=0.5, area_frac=0.5):
    return lambda x: _image_section(x, x_frac, y_frac, area_frac)


def _image_section(image, x_frac=0.5, y_frac=0.5, area_frac=0.5):
    x, y = image.shape
    x_mid = int(x * x_frac)
    y_mid = int(y * y_frac)
    x_deviation = int(.5 * x * np.sqrt(area_frac))
    y_deviation = int(.5 * y * np.sqrt(area_frac))
    return image[(x_mid - x_deviation):(x_mid + x_deviation), (y_mid - y_deviation):(y_mid + y_deviation)]


def raw_to_png(folder_path, cutting_fn=None, align=True):
    target_path = os.path.join(folder_path, "png")
    os.makedirs(target_path, exist_ok=True)
    reader = RawReader()
    images_list = reader.read_folder(folder_path)
    images_list = _sort_by_time(images_list)
    if align:
        images_list = _align_images(images_list)
    for i, image in enumerate(images_list):
        file_name = os.path.join(target_path, str(i))
        image_data = _preprocess_image(image.data)
        if cutting_fn is not None:
            image_data = cutting_fn(image_data)
        cv2.imwrite(file_name + ".png", image_data)
        info_dict = image.metadata
        for key, item in list(info_dict.items()):
            if isinstance(item, bytes):
                del info_dict[key]
            elif isinstance(item, datetime):
                info_dict[key] = str(item)

        with open(file_name + ".json", 'w') as f:
            json.dump(info_dict, f)


def _align_images(list_of_images):
    print("Aligning images...")
    processor = ImageProcessor()
    for i, image in enumerate(tqdm(list_of_images)):
        processor.load_image(image.data)
        processor.align(preprocess=False)
        list_of_images[i].data = processor.result()
    return list_of_images


def _average_images(images, n_images=1, sliding=False):
    images = np.array([image.data for image in images])
    if sliding:
        if n_images == 1:
            factors = np.array([0.75, 0.25])
        elif n_images == 2:
            factors = [0.75, 0.16, 0.08]
        else:
            factors = [0.75] + [0.12 / n for n in range(1, n_images + 1)]
        return np.sum(np.array([factor * images.copy()[n:-(n_images - n + 1)] / np.sum(factors)
                                for factor, n in zip(factors, range(n_images, -1, -1))]),
                      axis=0)

    else:
        new_frame_number = images.shape[0] // n_images
        images = images[:new_frame_number * n_images]
        return np.mean(images.reshape(n_images, new_frame_number, images.shape[1], images.shape[2]), axis=0)


def raw_to_tiff(folder_path, overlay: List[str] = None, align=True, process=False):
    target_path = os.path.join(folder_path, "tiff")
    os.makedirs(target_path, exist_ok=True)
    reader = RawReader()
    images_list = reader.read_folder(folder_path, first_n=-1000)
    # images_list = _sort_by_time(images_list)  # deprecated
    if align:
        images_list = _align_images(images_list)
    images_list = [image.data for image in images_list]
    images_list = _average_images(images_list, 5, False)
    with tifffile.TiffWriter(os.path.join(target_path, 'growth.tif')) as stack:
        for i, raw in enumerate(images_list):
            if process:
                image = _preprocess_image(raw.data)
            else:
                image = img_as_uint(raw.data)
            if overlay is not None:
                metadata = raw.metadata
                image = _add_overlay(image, metadata, overlay, images_list[0].metadata)
            stack.write(image, contiguous=False)


def raw_tiff_to_video(folder_path, name=None, fps=15, align=True):
    target_path = os.path.join(folder_path, "video")
    os.makedirs(target_path, exist_ok=True)
    reader = RawReader()
    images_list = reader.read_folder(folder_path, every_nth=10)
    if align:
        images_list = _align_images(images_list)
    if name is None:
        name = 'video'
    video_shape = images_list[0].data.shape
    video = cv2.VideoWriter(os.path.join(target_path, name + '.avi'),
                            cv2.VideoWriter_fourcc(*"DIVX"),
                            fps, video_shape, isColor=False)
    print("Producing video...")
    for image in tqdm(images_list):
        image = _preprocess_image(image.data, local_hist=None)
        video.write(image)
    cv2.destroyAllWindows()
    video.release()
    print("Video *{}* was produced".format(name))


def raw_to_video(folder_path, destination_path=None, name=None, fps=None, overlay: List[str] = None, align=True):
    if destination_path is None:
        target_path = os.path.join(folder_path, "video")
    else:
        target_path = destination_path
    os.makedirs(target_path, exist_ok=True)
    reader = RawReader()
    images_list = reader.read_folder(folder_path, every_nth=20)
    images_list = _sort_by_time(images_list)
    if align:
        images_list = _align_images(images_list)
    size_tuple = (images_list[0].metadata['width'], images_list[0].metadata['height'])
    if name is None:
        name = 'video'
    if fps is None:
        fps = round(1 / (images_list[1].metadata['timestamp'] - images_list[0].metadata['timestamp']).total_seconds(),
                    2)
    vid = cv2.VideoWriter(os.path.join(target_path, name + '.avi'),
                          cv2.VideoWriter_fourcc(*"DIVX"),
                          fps, size_tuple, isColor=(overlay is not None))
    for raw in tqdm(images_list):
        image = _preprocess_image(raw.data)
        if overlay is not None:
            metadata = raw.metadata
            image = _add_overlay(image, metadata, overlay, images_list[0].metadata)

        vid.write(image)
    cv2.destroyAllWindows()
    vid.release()
    print("Video *{}* was produced".format(name))


def _sort_by_time(images_list):
    print("Sorting by time...")
    try:
        initial_time = images_list[0].metadata['timestamp']
        times = [image.metadata['timestamp'] - initial_time for image in tqdm(images_list)]
    except AttributeError as e:
        # if metadata not there, trust name natsorting during folder reading
        return images_list
    images_list = [x for (y, x) in sorted(zip(times, images_list), key=lambda pair: pair[0])]
    return images_list


def _add_overlay(image, metadata, overlay, first_image_metadata):
    image = np.stack((image,) * 3, axis=-1)
    dy = 20
    text_offset = 20
    for info in overlay:
        overlay_string = ' '
        text_offset = text_offset + dy
        if info == 'time':
            t = (metadata['timestamp'] - first_image_metadata['timestamp']).total_seconds()
            overlay_string = overlay_string + "time: " + str(round(t, 2)) + ' s'
        else:
            overlay_string = overlay_string + info + ': ' + str(round(metadata[info][0], 2)) + ' ' + str(
                metadata[info][1])
            image = cv2.putText(image, overlay_string,
                       fontFace=cv2.FONT_HERSHEY_COMPLEX,
                       org=(10, text_offset),
                       fontScale=0.75,
                       color=(0, 255, 0),
                       lineType=2)


def png_to_video(folder_path, input_dims=(1024, 1024), fps=10):
    target_path = os.path.join(folder_path, "video")
    os.makedirs(target_path, exist_ok=True)
    file_list = os.listdir(folder_path)

    video = cv2.VideoWriter(os.path.join(target_path, 'video.avi'), cv2.VideoWriter_fourcc(*'XVID'), fps, input_dims)
    for i, file in enumerate(natsorted(file_list)):
        if file.endswith('.png'):
            image = cv2.imread(os.path.join(folder_path, file))
            video.write(image)
    cv2.destroyAllWindows()
    video.release()


def png_to_tiff(folder_path):
    list_of_files = os.listdir(folder_path)
    list_of_files = natsort.natsorted(list_of_files)

    with tifffile.TiffWriter(os.path.join(folder_path, 'boundary detection.tif')) as stack:
        for filename in list_of_files:
            stack.write(
                io.imread(os.path.join(folder_path, filename)),
                contiguous=False
            )


if __name__ == '__main__':
    # HOW TO USE THIS CODE:
    #  1: Copy-paste directory path (keep the r in the beginning of string, otherwise Windows won't be happy!)
    FOLDER = r'E:\Cardiff LEEM\Raw_images\11032022\cooling'
    #  2: Choose appropriate function, depending on desired format.
    # raw_to_png(FOLDER)

    # 2.5: For the other ones, you can add an overlay of the info you want to print on the image. Most useful are:
    # FOV, Start Voltage, time (you can change color on line 74 or 114)

    # raw_tiff_to_video(FOLDER)
    # raw_to_tiff(FOLDER, overlay=['time', 'FOV'])
    raw_to_tiff(FOLDER, align=True, process=False)
