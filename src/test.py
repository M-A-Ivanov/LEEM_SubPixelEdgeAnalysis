import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import util
import os
from tqdm import tqdm
from edge_segmentation import EdgeContainer
from global_parameters import CANNY_SIGMA, MASK_SIZE
from global_paths import RESULTS_FOLDER
from image_processing import ImageProcessor

from systematic_stuff.fluctuations.boundary_analysis import distribution_analysis, fft_analysis, get_paper_results, \
    create_positions


def test_boundary_detection():
    processor = ImageProcessor()
    image = produce_edge(n_frames=1)
    processor.load_image(image)
    mask = processor.get_mask(20)
    mask = processor.cut_to_mask(mask)
    processor.global_hist_equal()
    processor.denoise_nlm(fast=False)
    processor.canny_edges(sigma=2)
    # processor.subpixel_edges(mask=None)
    # processor.clean_up(15)
    processor.figure_all()


def produce_edge(n_frames=800):
    size = (40, 320)
    frames = np.ones((n_frames, size[0], size[1]), dtype=np.uint8)*40
    frames[:, 0, 0] = 0
    frames[:, -1, -1] = 255
    frames[:, :int(size[0]/2), :] = 160
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(frames[0], cmap='gray')
    frames = filters.gaussian(frames, sigma=4)
    axs[1].imshow(frames[0], cmap='gray')
    for i in range(len(frames)):
        frames[i] = util.random_noise(frames[i], mode='gaussian', var=0.1)
        # frames[i] = util.random_noise(frames[i], mode='gaussian', var=0.002)
    axs[2].imshow(frames[0], cmap='gray')
    plt.show()
    if n_frames == 1:
        frames = np.squeeze(frames)
    return frames


def test_manual_masking(frames, target_folder, edges):
    processor = ImageProcessor()
    list_of_images = frames
    processor.load_image(list_of_images[0])
    # processor.align(preprocess=True)
    masks = [processor.get_mask(MASK_SIZE) for _ in edges]
    groupers_canny = [EdgeContainer() for _ in edges]
    # groupers_pino = [AllEdges() for _ in edges]
    print("Found images: {}".format(len(list_of_images)))
    for i, image in enumerate(tqdm(list_of_images)):
        # time = round((image.metadata['timestamp'] - first_image_time).total_seconds(), 3)
        try:
            processor.load_image(image)
            # processor.align(preprocess=True)
            for original_mask, grouper_canny in zip(masks, groupers_canny):
                mask = processor.cut_to_mask(original_mask)
                processor.global_hist_equal(mask)
                processor.denoise_bilateral()
                processor.canny_devernay_edges(mask=mask, sigma=CANNY_SIGMA)
                processor.clean_up(15)
                grouper_canny.append_edges(processor.edge_result())
                processor.revert(5)
                # processor.subpixel_edges()
                # grouper_pino.append_edges(processor.edge_result())
                # processor.revert(n_steps=4)
        except Exception as e:
            print("Frame {} failed with: {}".format(i, e))
    for edge, grouper_canny, mask in zip(edges, groupers_canny, masks):
        print(edge)
        path = os.path.join(target_folder, edge)
        os.makedirs(path, exist_ok=True)
        grouper_canny.save_coordinates(path, suffix="canny_devernay")


if __name__ == '__main__':
    target_folder = os.path.join(RESULTS_FOLDER, 'test')
    os.makedirs(target_folder, exist_ok=True)
    edge = 'test_edge2'
    images = produce_edge(n_frames=500)
    test_manual_masking(images, target_folder, [edge])
    res_path = os.path.join(target_folder, edge)
    create_positions(results_path=res_path, savefig=res_path)

    sigma = distribution_analysis(res_path, adjusted=False)
    beta = fft_analysis(res_path, adjusted=False, full=True)
    get_paper_results(res_path, beta=beta, sigma=sigma)
