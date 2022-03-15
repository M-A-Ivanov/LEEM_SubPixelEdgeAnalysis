import json
import os

import numpy as np

from data_analysis import PaperAnalysis
from global_paths import RESULTS_FOLDER
from systematic_stuff.fluctuations.boundary_analysis import _all_regions, _all_edges


def open_json(region, edge):
    with open(os.path.join(RESULTS_FOLDER, region, edge, "results.json")) as json_file:
        return json.load(json_file)


def average_results(list_of_results):
    paper = PaperAnalysis()
    sigmas = np.array([result["sigma"][0] for result in list_of_results])
    betas = np.array([result["beta"][0] for result in list_of_results])
    lambdas = np.array([result["d_lambda"][0] for result in list_of_results])
    C_ms = np.array([paper.get_C0_from_c(paper.get_c_from_data(beta, sigma)) for beta, sigma in zip(betas, sigmas)])*0.1  # to Angstrom
    avg_C_ms = np.average(C_ms)
    # dS = paper.get_entropy_from_C0(avg_C_ms)
    dS = np.array([paper.get_entropy_from_C0(C_m) for C_m in C_ms])
    overall_results = dict()
    overall_results['sigma'] = (np.average(sigmas), np.std(sigmas), "1/nm")
    overall_results['beta'] = (np.average(betas), np.std(betas), "meV/nm^2")
    overall_results['Cm'] = (round(avg_C_ms, 7), np.std(C_ms), "meV/A")
    overall_results['d_lambda'] = (round(np.average(lambdas), 4), np.std(lambdas), "eV/A")
    overall_results['dS_A2'] = (np.average(dS), np.std(dS), "eV/A^2")
    dS = dS * 16. * 1.e3
    overall_results['dS_1x1'] = (np.average(dS), np.std(dS), "meV/(1x1)")
    print(overall_results)
    with open(os.path.join(RESULTS_FOLDER, "overall_results.json"), 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    list_of_results = []
    for region in _all_regions():
        for edge in _all_edges(region):
            list_of_results.append(open_json(region, edge))

    average_results(list_of_results)

