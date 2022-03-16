import json
import os
import pickle
import numpy as np
from skimage.io import imsave

from data_analysis import DistributionResults, PaperAnalysis, FFTResults
from global_paths import RESULTS_FOLDER, SRC_FOLDER, REGION, load_pickle, TARGET_FOLDER
from global_parameters import pixel
from edge_processing import EdgeAnalyzer

from raw_reader import RawReader

from systematic_stuff.convenience_functions import _preprocess_image
from systematic_stuff.fluctuations.boundary_detection import FluctuationsDetector


def _all_regions():
    return ['3microns, 0.2s, r' + str(n) for n in np.arange(1, 7)]
    # return ['3microns, 0.2s, r6']


def _all_edges(region):
    # return [path for reg in regions for path in os.listdir(os.path.join(RESULTS_FOLDER, reg))]
    return [edge for edge in os.listdir(os.path.join(RESULTS_FOLDER, region)) if "excl" not in edge]


def make_directories(region_names: list, edge_names: list):
    for reg in region_names:
        for e in edge_names:
            dir_path = os.path.join(RESULTS_FOLDER, reg, e)
            os.makedirs(dir_path, exist_ok=True)


def trim_array(arr, Hbounds, Wbounds):
    arr = arr[np.where(np.logical_and(arr[:, 0] >= Hbounds[0], arr[:, 0] <= Hbounds[1]))]
    arr = arr[np.where(np.logical_and(arr[:, 1] >= Wbounds[0], arr[:, 1] <= Wbounds[1]))]
    return arr


def create_positions(results_path, adaptive=True, savefig=None):
    """Algorithm 2: Transform coordinates of boundary to offsets wrt mean boundary position.
                    * Coordinates of shape: list [frame, array(n_points, 2)] is loaded.
                    * Flip all coordinates from (y, x) to (x, y) for consistency in calculations
                    * Create perpendiculars to coordinates and save perpendiculars plot image
                        - A 'perpendicular' object is defined by two points (mid and second). 'mid' refers to
                            the point on the boundary used. 'second' refers to a calculated point, such that the line
                            between the two points is perpendicular to the straight line fit of the surrounding N edges
                            coordinates. The Perpendicular object contains all point projection methods that translate
                            coordinates to points on the Perpendicular - e.g. offsets.append
                        - Save all perpendiculars in shape array(frame, mid, second).
                    * For each frame, find a point on each perpendicular that best represents original coordinates using
                        available projection methods: 'weno', 'interpolation', or 'simple'.
                    * Save offsets in shape array(n_perpendiculars, frame)
                    * In addition, save relevant information from the process in .txt file
                """

    main_coords = os.path.join(results_path, "coordinates_canny_devernay.pickle")
    c = load_pickle(main_coords)

    anal = EdgeAnalyzer(c, low_memory=True)
    if adaptive is True:
        anal.get_perpendiculars_adaptive(edge_frame=0, every_point=True, pixel_average=32, savefig=savefig)
    else:
        anal.get_perpendiculars_simple(edge_frame=0, pixel_average=5, savefig=savefig)
    picklepath_perp = os.path.join(results_path, "perpendiculars.pickle")
    with open(picklepath_perp, "wb") as f:
        pickle.dump(anal.perpendiculars_to_save(), f)
    positions, info = anal.get_edge_variations(savefig=None)
    edge_length = anal.get_length()
    picklepath = os.path.join(results_path, "offsets_original.pickle")
    with open(picklepath, "wb") as f:
        pickle.dump((positions, edge_length), f)
    infopath = os.path.join(results_path, "detection_info.json")
    with open(infopath, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def create_positions_adjusted(results_path, region, adjustment: str = "tanh", savefig=None, debug=False):
    main_coords = os.path.join(results_path, "coordinates_canny.pickle")
    c = load_pickle(main_coords)

    anal = EdgeAnalyzer(c)
    anal.get_perpendiculars_adaptive(edge_frame=0, every_point=True, pixel_average=32, savefig=savefig)
    picklepath_perp = os.path.join(results_path, "perpendiculars.pickle")

    with open(picklepath_perp, "wb") as f:
        pickle.dump(anal.perpendiculars_to_save(), f)

    positions, info_rough = anal.get_edge_variations(savefig=None)
    if adjustment == "tanh":
        adjusted, info_adj = anal.adjust_edge_variations(rough_offsets=positions,
                                                         region_path=os.path.join(SRC_FOLDER, region),
                                                         max_dist=20)

    else:
        print("adjustment type {} not understood. Use 'tanh'".format(adjustment))
        return
    edge_length = anal.get_length()
    picklepath_adj = os.path.join(results_path, "offsets_adjusted.pickle")
    picklepath_ori = os.path.join(results_path, "offsets_original.pickle")
    with open(picklepath_adj, "wb") as f:
        pickle.dump((adjusted, edge_length), f)
    with open(picklepath_ori, "wb") as f:
        pickle.dump((positions, edge_length), f)
    infopath = os.path.join(results_path, "detection_info.json")
    info = {**info_rough, **info_adj}
    with open(infopath, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def adjust_rough_offsets(results_path):
    main_coords = os.path.join(results_path, "coordinates_canny.pickle")
    c = load_pickle(main_coords)

    anal = EdgeAnalyzer(c)
    picklepath = os.path.join(results_path, "offsets_original.pickle")
    with open(picklepath, "rb") as f:
        positions, lengths = pickle.load(f)
        
    perpspath = os.path.join(results_path, "perpendiculars.pickle")
    with open(perpspath, "rb") as f:
        perps = pickle.load(f)
    anal.perpendiculars_to_load(perp_arr=perps)
    adjusted, info_adj = anal.adjust_edge_variations(rough_offsets=None,
                                                     region_path=os.path.join(SRC_FOLDER, region),
                                                     max_dist=10, adjustment_type="tanh")
    edge_length = anal.get_length()
    picklepath_adj = os.path.join(results_path, "offsets_adjusted.pickle")
    infopath = os.path.join(results_path, "detection_info_adj_only.json")
    with open(picklepath_adj, "wb") as f:
        pickle.dump((adjusted, edge_length), f)
    with open(infopath, 'w', encoding='utf-8') as f:
        json.dump(info_adj, f, ensure_ascii=False, indent=4)
    return adjusted


def distribution_analysis(results_path, adjusted=False):
    anal = DistributionResults(results_path, original=not adjusted)
    mean, std = anal.get_analysis()
    print("mean = {} \n sigma = {}".format(round(mean * pixel, 5), round(std * pixel, 2)))
    return std * pixel


def fft_analysis(results_path, adjusted=False, full=True):
    anal = FFTResults(results_path, original=not adjusted)
    return anal.get_analysis(full=full)


def get_correlations(results_path, fps, adjusted=False):
    anal = FFTResults(results_path, original=not adjusted)
    return anal.get_fourier_correlation(fps)


def get_paper_results(results_path, beta, sigma):
    paper = PaperAnalysis()
    results = {'beta': (beta, "meV/nm^2"),
               'sigma': (sigma, "1/nm")}
    c = paper.get_c_from_data(beta, sigma)  # beta : meV/nm^2, sigma nm-1
    # c = paper.get_c_from_data(190, np.sqrt(78.1))
    c0 = paper.get_C0_from_c(c)  # meV/nm
    print("C_0 = {} eV/A".format(round(c0 * 1e-4, 5)))
    results['Cm'] = (round(c0 * 1e-4, 5), "eV/A")
    stress = paper.get_stress_from_C0(c0)
    print('delta_lambda = {} eV/A'.format(round(stress, 3)))
    results['d_lambda'] = (round(stress, 5), "eV/A")
    S = paper.get_entropy_from_C0(c0)
    print('delta_S = {} eV/A^2 or {} meV/(1x1)'.format(round(S, 10), round(S * 16 * 1e3, 5)))
    results['dS_A2'] = (round(S, 10), "eV/A^2")
    results['dS_1x1'] = (round(S * 16 * 1e3, 7), "meV/(1x1)")
    import json
    path = os.path.join(results_path, "results.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def draw_analyzed_edge(results_path, region):
    read = RawReader()
    seed = 0
    source_folder = os.path.join(SRC_FOLDER, region)
    image = _preprocess_image(read.read_single(source_folder, frames=seed).data)
    image = np.stack((image,) * 3, axis=-1)
    color = np.array([0, 255, 0])
    alpha = 0.5
    edge_coords = load_pickle(os.path.join(results_path, 'coordinates_canny_devernay.pickle'))[seed]
    edge_coords = np.rint(edge_coords).astype(int)
    image[edge_coords[:, 0], edge_coords[:, 1]] = (1 - alpha) * image[
        edge_coords[:, 0], edge_coords[:, 1]] + alpha * color
    imsave(os.path.join(results_path, "analyzed_edge.png"), image)


if __name__ == "__main__":
    do_detection = 0
    do_transform = 1
    do_analysis = 1
    regions = [REGION]
    # region = _all_regions()
    # edges = EDGE
    edges = ["edge 1", "edge 2", "edge 3"]
    # edges = _all_edges(regions)
    make_directories(regions, edges)

    for region in regions:
        if do_detection:
            detector = FluctuationsDetector(os.path.join(SRC_FOLDER, region), TARGET_FOLDER, edges)
            detector.many_edge_canny_devernay_detection(load_masks=False)
        if do_transform:
            for edge in edges:
                res_path = os.path.join(RESULTS_FOLDER, region, edge)
                create_positions(results_path=res_path, savefig=res_path, adaptive=True)
        if do_analysis:
            for edge in edges:
                res_path = os.path.join(RESULTS_FOLDER, region, edge)
                sigma = distribution_analysis(res_path, adjusted=False)
                beta = fft_analysis(res_path, adjusted=False, full=False)
                # get_correlations(res_path,  fps=20, adjusted=False)
                draw_analyzed_edge(res_path, region)
                get_paper_results(res_path, beta=beta, sigma=sigma)
