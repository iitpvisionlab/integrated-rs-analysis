import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.interpolate import interp1d
import glob
import os
import argparse
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        "Generate sentinel and sequoia synthetic data from hyperspectral data mask for one image"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="path to hyperspectral data"
    )
    parser.add_argument(
        "-o1", "--output_sequoia", required=True, help="sequoia output folder path"
    )
    parser.add_argument(
        "-o2", "--output_sentinel", required=True, help="sentinel output folder path"
    )

    return parser.parse_args()


def delete_bad_rows(x):
    # subscript to get rid of unnecessary wavelengths
    indices_x = np.argwhere((x[4:60] < -1e-6) | (x[4:60] > 1))[:, 1]

    indices = list(set(indices_x))

    x = np.delete(x, indices, axis=1)
    return x


def delete_inf_rows(x):
    # subscript to get rid of unnecessary wavelengths
    indices_x = np.argwhere(np.isinf(x) | np.isnan(x))[:, 1]

    indices = list(set(indices_x))

    x = np.delete(x, indices, axis=1)
    return x


def fix_bad_rows(x):
    x[x < 0] = 0
    x[x > 1] = 1

    tmp = x.sum(axis=0)

    indices = np.argwhere((tmp < 0.1))

    x = np.delete(x, indices, axis=1)

    return x


def change_wavelengths(old_wavelengths, old_vals, new_wavelengths, assume_sorted=False):
    from scipy.interpolate import interp1d

    """recalculate spectrum from ond wavelengths to other
    Args:
        old_wavelengths (np.ndarray): wavelengths (let's say in nm)
        old_vals (np.ndarray): corresponding values
        new_wavelengths (np.ndarray): (let's say in nm)

    Returns:
        new_vals (np.ndarray)
    """

    func = interp1d(
        old_wavelengths,
        old_vals,
        axis=0,
        fill_value=0,
        bounds_error=False,
        assume_sorted=assume_sorted,
    )
    new_vals = func(new_wavelengths)
    return new_vals


def get_sequoia_sensitivities(wvs, base_root=""):
    files_path = sorted(Path(base_root).glob("sequoia_sensor/*.json"))

    sens_dict = {}
    for file_path in files_path:
        print(file_path)
        with open(file_path, "r") as fp:
            data = json.load(fp)

        x = data["x"]
        y = data["y"]

        file_path = str(file_path)
        name = file_path[file_path.rfind("_") + 2 : file_path.find(".")]
        sens_dict[name] = change_wavelengths(x, y, wvs)
    return sens_dict


def get_satellite_sensitivities(wvs, bands, satellite="Sentinel2A", base_root=""):
    AVAILABLE_SATELLITES = ["EarthObserving1", "Sentinel2A", "LandsatOLI8"]
    assert (
        satellite in AVAILABLE_SATELLITES
    ), f"'{satellite}' not in {AVAILABLE_SATELLITES}"
    file = str(Path(base_root) / "satellite_sensitivities" / f"{satellite}.csv")

    s2a = pd.read_csv(file)

    funcs = {}
    for name in s2a.columns[1:]:
        wavelengths = s2a["SR_WL"]
        vals = s2a[name]
        if not name in bands:
            continue
        funcs[name] = change_wavelengths(wavelengths, vals, wvs)

    return funcs


def interpolate_1d_2d_data(wavelengths, data, new_wvs, step=250000):
    check_i = 0
    n = data.shape[1]

    new_data = np.zeros((len(new_wvs), n))
    while check_i != n:
        end_i = min(check_i + step, n)
        new_data[:, check_i:end_i] = change_wavelengths(
            wavelengths, data[:, check_i:end_i], new_wvs, True
        )
        check_i = end_i
    return new_data


def _main(input, output_sequoia, output_sentinel):
    assert os.path.exists(output_sequoia), "output folder does not exist"
    assert os.path.exists(output_sentinel), "output folder does not exist"

    # CA - Coastal Aerosol, B - Blue, G - Green, R - Red, REGK - Red Edge K, NIRB - NIR Broad, NIRN - NIR Narrow
    bands_range = {
        "CA": [430, 450],
        "B": [450, 510],
        "G": [530, 590],
        "R": [640, 670],
        "REG1": [690, 710],
        "REG2": [730, 750],
        "REG2": [770, 790],
        "NIRB": [780, 880],
        "NIRN": [850, 880],
        "SWIR1": [1570, 1650],
        "SWIR2": [2110, 2290],
    }

    wavelengths = []

    with open(
        "MultispectralData/f080814t01p00r19/f080814t01p00r19/f080814t01p00r19rdn_c.spc",
        "r",
    ) as f:
        lines = f.readlines()
        for line in lines:
            wv = float(line[: line.find("\t")])
            wavelengths.append(wv)

    new_wvs = np.concatenate(
        (np.arange(430, 921, 2), wavelengths[wavelengths.index(927.92145) :])
    )

    data_paths = glob.glob(f"{input}/*")

    for data_path in tqdm(data_paths):
        data = np.load(data_path)
        head, tail = os.path.split(data_path)

        nrows = data.shape[1]
        ncols = data.shape[2]

        # reshape data
        data = data.reshape(224, -1)
        data = delete_bad_rows(data)
        data = delete_inf_rows(data)
        data = fix_bad_rows(data)
        data = interpolate_1d_2d_data(wavelengths, data, new_wvs)
        # print(data.shape)
        # print(data[data < 0])

        bands = [
            "Band 1",
            "Band 2",
            "Band 3",
            "Band 4",
            "Band 5",
            "Band 6",
            "Band 7",
            "Band 8",
            "Band 8A",
        ]
        sentinel_sensitivities = get_satellite_sensitivities(
            new_wvs, bands, satellite="Sentinel2A"
        )
        sentinel_data = np.zeros((len(sentinel_sensitivities), data.shape[1]))

        # get solar radiance
        sys.path.append("../satspectra")
        from illumination.spectrl2 import solar
        from utils.config import CONFIG

        zenith = 0
        solar_spectrum = solar(CONFIG, apparent_zenith=zenith)
        solar_spectrum = change_wavelengths(
            CONFIG["wavelengths"], solar_spectrum, new_wvs
        )
        # solar spectrum [0, 1] normalisation
        solar_spectrum /= solar_spectrum.max()

        # generate synthetic data
        for i, (name, spectrum) in enumerate(sentinel_sensitivities.items()):
            spectrum *= solar_spectrum
            result = (spectrum @ data) / spectrum.sum()
            sentinel_data[i] = result
        # save
        np.save(f"{output_sentinel}/sentinel_{tail}", sentinel_data)

        sequoia_sensitivities = get_sequoia_sensitivities(new_wvs)
        sequoia_data = np.zeros((len(sequoia_sensitivities), data.shape[1]))

        for i, (name, spectrum) in enumerate(sequoia_sensitivities.items()):
            spectrum *= solar_spectrum
            result = (spectrum @ data) / spectrum.sum()
            sequoia_data[i] = result
        np.save(f"{output_sequoia}/sequoia_{tail}", sequoia_data)


if __name__ == "__main__":
    args = parse_args()
    _main(args.input, args.output_sequoia, args.output_sentinel)
