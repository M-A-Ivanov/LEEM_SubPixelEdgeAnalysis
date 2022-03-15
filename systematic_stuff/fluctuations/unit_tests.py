import unittest
import random

from data_analysis import ExperimentalFFTResults
import numpy as np

from global_parameters import pixel


class FFTAnalysisTest(unittest.TestCase):
    @staticmethod
    def _generate_data(n_frames, length):
        mu, sigma = random.uniform(-1, 1), random.uniform(8, 16)

        data = np.random.normal(loc=mu, scale=sigma, size=(n_frames, length))
        print(data.shape)
        return data

    def setUp(self):
        n_frames = 500
        length = 300
        self.fft_a = ExperimentalFFTResults(self._generate_data(n_frames, length), length=length*pixel)

    def test_fft(self):
        fft_array = self.fft_a.get_fft()
        n = fft_array.shape[0]
        dx = np.arange(n)
        q = 2 * dx * np.pi / self.fft_a.length
        y_x_t = np.empty_like(fft_array, dtype=np.complex128)
        for i in range(fft_array.shape[1]):  # for each frame
            for j in range(n):  # for each mode
                y_x_t[j, i] = np.mean(np.prod((fft_array[:, i], np.exp(1j * q[j] * dx)), axis=0))
        assert np.isclose(y_x_t, self.fft_a.positions).all()




if __name__ == '__main__':
    FFTAnalysisTest()