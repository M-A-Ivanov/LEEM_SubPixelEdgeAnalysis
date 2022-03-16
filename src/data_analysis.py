import os
import pickle
from typing import List

import numpy as np
from scipy import stats, optimize, spatial, fft, fftpack, signal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

from global_paths import DANIDATA_PATH, RESULTS_FOLDER, REGION, EDGE
from global_parameters import pixel
import json


class Results:
    def __init__(self, results_path, original=False):
        self.results_path = results_path
        self.positions, self.length = self.load_data(results_path, original)
        self.remove_drift()

    def load_data(self, results_path, original=False):
        if results_path is None:
            return None, None
        if original:
            picklepath = os.path.join(results_path, "offsets_original.pickle")
        else:
            picklepath = os.path.join(results_path, "offsets_adjusted.pickle")
        with open(picklepath, "rb") as f:
            positions, lengths = pickle.load(f)
        length = np.average(lengths)*pixel
        positions = pixel*positions
        return positions, length

    def remove_drift(self):
        if self.positions is None:
            return
        for i, perpendicular_detection in enumerate(self.positions):
            self.positions[i] = self.positions[i] - np.average(self.positions[i])

    def get_analysis(self, *args):
        raise NotImplementedError

    def plot_offsets(self):
        plt.figure()
        for n in range(10):
            plt.plot(self.positions[:, n]+5*n)
        plt.ylim(-5, 50)
        plt.show()
        plt.close()

    def analyze_corrections(self):
        original, _ = self.load_data(self.results_path, original=True)
        corrected, _ = self.load_data(self.results_path, original=False)
        corrections = (corrected - original)*pixel
        plt.hist(corrections)
        plt.show()
        # n, bin_edges = np.histogram(corrections, bins=20, density=True)
        # x = .5 * (bin_edges[1:] + bin_edges[:-1])
        # gauss_x = np.linspace(np.min(x), np.max(x), 1000)
        # # mean, std = stats.norm.fit(data)
        # _, mean, std = self.fit_gauss(x, n)
        # gauss = stats.norm.pdf(gauss_x, mean, std)
        # plt.figure()
        # plt.plot(gauss_x, gauss, color='b')
        # plt.scatter(x, n, facecolors='none', edgecolors='k', marker='o', alpha=0.8)


class FFTResults(Results):
    def __init__(self, results_path, original=False):
        super(FFTResults, self).__init__(results_path, original)

    @staticmethod
    def fourier_transform(y_x_t):
        y_q = np.zeros_like(y_x_t, dtype=np.complex128)
        if y_x_t.ndim > 1:
            for frame in range(y_x_t.shape[-1]):
                y_q[:, frame] = fftpack.fft(y_x_t[:, frame])
        else:
            y_q = fftpack.fft(y_x_t)
        return y_q[1:y_q.shape[0]//2, ]

    def manual_fft(self, y_x_t):
        n = y_x_t.shape[0]
        dx = np.arange(n)
        q = 2 * dx * np.pi / self.length
        y_q = np.empty_like(y_x_t, dtype=np.complex128)
        for i in range(y_x_t.shape[1]):  # for each frame
            for j in range(n):  # for each mode
                y_q[j, i] = np.sum(np.prod((y_x_t[:, i], np.exp(-1j*q[j]*dx)), axis=0))
        return y_q

    def get_fft(self):
        # return (1 / np.sqrt(self.length)) * self.fourier_transform(self.positions)
        return (1 / np.sqrt(self.length)) * self.manual_fft(self.positions)

    def get_y_q_msa(self):
        array_fft = self.get_fft()
        y_q_msa = np.mean(np.square(np.absolute(array_fft)), axis=1)

        n = np.arange(1, len(y_q_msa))
        y_q_msa = self._apply_gauss_factor(y_q_msa)
        q = (2 * np.pi * n) / self.length

        return n, q, y_q_msa[1:]

    def _apply_gauss_factor(self, y):
        real_q = 2 * np.pi * np.arange(len(y)) / self.length
        gaussian_factor_ampl = np.exp(-(30 * pixel * pixel) * real_q * real_q / 8)
        return y*gaussian_factor_ampl

    def get_analysis(self, full=True):
        print("L={}".format(round(self.length, 2)))

        n, q, y_q_msa = self.get_y_q_msa()
        if not full:
            n = n[2:15]
            q = q[2:15]
        proportional_q = (1./q)**2
        slope, intercept, _, _, _ = stats.linregress(proportional_q, y_q_msa[n-1])
        print("slope = {} \n intercept = {}".format(round(slope, 5), round(intercept, 5)))
        plt.figure()
        plt.plot(proportional_q, (slope * proportional_q + intercept), color='b')
        selected_y = y_q_msa[n]
        plt.scatter(proportional_q, selected_y, facecolors='none', edgecolors='k', marker='o', alpha=0.8)

        if full:
            suffix = "_full"
        else:
            suffix = "_partial"
        plt.savefig(os.path.join(self.results_path, 'fft_fig_weno{}.png'.format(suffix)), dpi=1000, transparent=False)
        plt.close()

        k_B = 8.617e-2
        T = 550 + 273.15
        beta = k_B * T / (self.length * slope)
        print("beta = {}".format(round(beta, 3)))
        return beta

    def _q_selector(self, wavelength):
        return int(self.length/(2*np.pi*wavelength))

    def get_fourier_correlation(self, fps):
        def apply_correlation_common(arr, n_lags=20):
            lags = np.arange(1, n_lags)
            G = np.zeros(n_lags)
            for i, lag in enumerate(lags):
                G[i+1] = np.average(np.square(np.absolute(np.subtract(arr[:-lag], arr[lag:]))))
            return G[1:]

        def apply_correlation_alt(arr, n_lags=20):
            lags = np.arange(0, n_lags)
            G = np.ones(n_lags)
            for i, lag in enumerate(lags):
                G[i] = 2*np.average(np.absolute(np.square(arr))) - 2*np.average(np.roll(arr, lag)*np.flip(arr))
            return G

        import pandas as pd

        def apply_correlation_pandas(arr, n_lags=20):
            to_series = pd.Series(arr)
            correlations = np.empty(n_lags)
            for i in range(n_lags):
                correlations[i] = to_series.corr(to_series.shift(i+1), method="pearson")
            return correlations

        def apply_correlation_acf(arr, n_lags=20):
            return acf(arr, fft=False, nlags=n_lags)

        def _exp_fit(t, tau):
            return np.exp(-t/tau)
        time = (1/fps)*np.arange(self.positions.shape[1])
        # array_fft = self.manual_fft(self.positions)
        array_fft = self.get_fft()
        plt.figure()
        G = np.apply_along_axis(apply_correlation_common, 1, array_fft, **{"n_lags": 200})
        G = G/np.max(G)
        for n in np.linspace(1, 10, 10, dtype=np.int):
            plt.plot(time[:G[n].size], G[n, ], label="n={}".format(n))
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(self.results_path, 'correlation_fig.png'), dpi=1000, transparent=False)
        plt.close()

    def get_time_correlation(self, y=30):
        analysed_position = self.positions[y, :]
        n_lags = 100
        correlation = np.zeros(n_lags-1)
        for lag in range(1, n_lags):
            correlation[lag-1] = np.average(np.square((np.subtract(analysed_position[lag], analysed_position[0]))))
        # correlation = np.square(acf(analysed_position, fft=False, nlags=50, adjusted=False))
        plt.plot(np.arange(len(correlation)), correlation)
        plt.show()


class ExperimentalFFTResults(FFTResults):
    def __init__(self, offsets, length):
        super(ExperimentalFFTResults, self).__init__(results_path=None)
        self.positions = offsets
        self.length = length


class DistributionResults(Results):
    def __init__(self, results_path, original=False):
        super(DistributionResults, self).__init__(results_path, original)

    def fit_gauss(self, x, y):
        def gauss(x, *p):
            A, mu, sigma = p
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        p0 = [0.1, 1, 1]  # (A, mu and sigma above)
        coeff, var_matrix = optimize.curve_fit(gauss, x, y, p0=p0)
        return coeff

    def get_distribution(self):
        data = self.positions.flatten()
        bins = int(2*(np.max(data) - np.min(data)))
        data = data
        n, bin_edges = np.histogram(data, bins=bins, density=True)
        x = .5 * (bin_edges[1:] + bin_edges[:-1])
        gauss_x = np.linspace(np.min(x), np.max(x), 1000)
        # mean, std = stats.norm.fit(data)
        _, mean, std = self.fit_gauss(x, n)
        gauss = stats.norm.pdf(gauss_x, mean, std)
        return x, n, gauss_x, gauss, mean, std

    def get_analysis(self):
        x, n, gauss_x, gauss, mean, std = self.get_distribution()
        plt.figure()
        plt.plot(gauss_x, gauss, color='b')
        plt.scatter(x, n, facecolors='none', edgecolors='k', marker='o', alpha=0.8)
        # axis_lim = np.min((abs(np.min(x)), abs(np.max(x))))
        # if axis_lim > 15:
        #     axis_lim = 15
        axis_lim = 20
        plt.xlim(-axis_lim, axis_lim)
        # plt.ylim(0, 0.2)

        plt.savefig(os.path.join(self.results_path, 'distr_fig.png'), dpi=1000, transparent=False)
        # plt.show()
        plt.close()
        return mean, std


class ExperimentalDistributionResults(DistributionResults):
    def __init__(self, offsets, length):
        super(ExperimentalDistributionResults, self).__init__(results_path=None)
        self.positions = offsets
        self.length = length


class PaperAnalysis:
    @staticmethod
    def get_c_from_data(beta, sigma):
        k_B = 8.617e-2  # meV/K
        T = 550 + 273.15  # K
        # T = 1135  # Si

        c = (k_B * k_B * T * T) / (8 * beta * sigma ** 4)
        # C_0 = (w * w / np.pi ** 2) * c / (1 + np.tan(0. * np.pi / 2) ** 2)
        return c  # meV/nm^3

    def get_C0_from_c(self, c):
        w = 165  # nm, width of terraces
        return c * ((w * w) / (np.pi * np.pi))  # meV/nm

    def get_stress_from_C0(self, c0):
        M = 0.53  # ev/A^3, Young's modulus
        # M = 1  # Si
        ni = 0.31  # Poisson ratio
        # ni = 0.27  # Si
        c0 = c0 * 1e-4  # meV/nm to eV/A

        stress = np.sqrt(np.pi * c0 * M / (1 - ni * ni))

        return stress

    def get_entropy_from_C0(self, c0):
        c0 = c0 * 1e-4  # meV/nm to eV/A
        dS_over_c0 = 1.1905 * 1e-4  # K-1.micron-1 to K-1.A-1
        return c0 * dS_over_c0

    @staticmethod
    def do_hannon_fit():
        L = 1650.  # angstrom
        path = os.path.join(DANIDATA_PATH, "danidata.csv")
        with open(path, 'r', encoding='utf-8-sig') as csv_file:
            data = np.genfromtxt(csv_file, delimiter=',', encoding="utf8")
            T = data[:, 0]
            tanp = data[:, 1]

        def hannon_eqn(tan_p, dS, Cm, Cd, Cr, const):
            p = (2. / np.pi) * np.arctan(tan_p)
            return (2. / dS) * (-(np.pi * Cm * tan_p / L)
                                - (np.pi * Cd / (2 * L * L)) * (1. / (np.cos(p * np.pi / 2) ** 2))
                                - (16 * Cr / (L ** 3)) * ((1 / ((1 - p) ** 3)) - (1 / ((1 + p) ** 3)))) + const

        # guess = [0.00031 * 0.12 * 2 * np.pi / 3000, 0.00031, 0.81, 8.6, 10]  # Hannon's
        guess = [4e-7, 0.00433, 1, 10, 10]
        param_bounds = (np.array([1e-9, 1e-4, 1e-2, 1e-1, 1e-1]),
                        np.array([1e-5, 1e-1, 10, 100, 100]))

        coeff, var_matrix = optimize.curve_fit(hannon_eqn, tanp, T, p0=guess, bounds=param_bounds)
        print(coeff)
        print('Error is {}'.format(np.sqrt(np.diag(var_matrix))))
        tanp_cont = np.linspace(1, -4, 500)
        fit = hannon_eqn(tanp_cont, *coeff)
        fit_approx = hannon_eqn(tanp_cont, coeff[0], coeff[1], 0, 0, 0)

        linear_region = (-18, 12)
        linear_region_T = T[np.where((T > linear_region[0]) & (T < linear_region[1]))]
        linear_region_tanp = tanp[np.where((T > linear_region[0]) & (T < linear_region[1]))]

        slope, intercept, _, _, _ = stats.linregress(linear_region_T, linear_region_tanp)
        print("slope = {}, intercept = {}, dS/C2 = {}".format(slope, intercept, 1e4 * slope * (-2 * np.pi / L)))
        fig = plt.figure(figsize=(5, 6))
        ax = fig.add_subplot(111)
        ax.set_aspect(25)
        ax.set_ylabel(r"$\tan(\dfrac{p\pi}{2})$")
        ax.set_xlabel(r"$T-T_{0} \; \; (^{\circ}$C)")
        ax.axline([0, intercept], slope=slope, color='k', linestyle="--", zorder=4)
        # hannon_coeff = [0.00031 * 0.12 * 2 * np.pi / 3000, 0.00031, 0.81, 8.6, 3.4]
        # print(hannon_coeff)
        # fit2 = hannon_eqn(tanp_cont, *hannon_coeff)
        # plt.plot(fit2, tanp_cont, 'k')
        ax.scatter(T, tanp, color='r', zorder=3)
        ax.axhline(linewidth=0.4, color='gray', zorder=1)
        ax.axvline(linewidth=0.4, color='gray', zorder=1)
        ax.plot(fit, tanp_cont, 'gray', zorder=2)
        # ax.plot(fit_approx, tanp_cont, 'red', zorder=2)
        plt.xlim((min(fit), max(fit)))
        plt.ylim((-4, 1))
        # plt.show()
        # plt.close()
        os.makedirs(os.path.join(RESULTS_FOLDER, "for_figures"), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_FOLDER, "for_figures", "hannon_fit.png"),
                    dpi=1000, transparent=True,
                    # bbox_inches='tight',
                    )
        # np.savetxt("E:\\results\\output.csv", np.column_stack((fit, tanp_cont)), delimiter=",")

    @staticmethod
    def plot_coverage_proportions():
        path = os.path.join(DANIDATA_PATH, "coverage_proportions.csv")
        with open(path, 'r', encoding='utf-8-sig') as csv_file:
            data = np.genfromtxt(csv_file, delimiter=',', encoding="utf8")
            T = data[:, 0] + 47
            sixbysix = data[:, 1]
            eightbytwo = data[:, 2]

        plt.figure(figsize=(8, 6))
        plt.scatter(T, eightbytwo, edgecolors='k', facecolors='none')
        plt.scatter(T, sixbysix, color='k')
        plt.xlabel(r"$T \; \; (^{\circ}$C)")
        plt.ylabel(r"Phase coverage (%)")

        os.makedirs(os.path.join(RESULTS_FOLDER, "for_figures"), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_FOLDER, "for_figures", "coverage_proportions.png"), dpi=1000, transparent=True)
        # plt.show()
        # plt.close()


class SummingUp:
    def __init__(self, results_path: str, regions: List = None):
        self.results = []
        if regions is None:
            self.regions = self._get_subfolders(results_path, "3microns, 0.2s")
        else:
            self.regions = regions
        # noinspection PyBroadException
        self.all_paths = [os.path.join(self._get_subfolders(os.path.join(results_path, region), "edge", join=True))
                          for region in self.regions]

    @staticmethod
    def _get_subfolders(path, common_str: str = None, join: bool = False):
        if common_str is not None:
            dirs = [x[0] for x in os.walk(path) if common_str in x[0]]
        else:
            dirs = [x[0] for x in os.walk(path)]
        if join:
            dirs = [os.path.join(path, dir) for dir in dirs]

        return dirs

    @staticmethod
    def _get_info_from_path(path):
        path, _ = os.path.split(path)
        rest, edge = os.path.split(path)
        _, region = os.path.split(rest)
        return [region, edge]

    def extract_results(self):
        for path in self.all_paths:
            with open(os.path.join(path, "results.json")) as f:
                results = json.load(f)
                results["info"] = self._get_info_from_path(path)
                self.results.append(results)


if __name__ == '__main__':
    """Currently, experiment with shortening the duration."""
    results_path = os.path.join(RESULTS_FOLDER, REGION, EDGE)
    anal = FFTResults(results_path, original=True)
    beta = anal.get_analysis(full=True)
