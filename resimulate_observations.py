# -*- coding: utf-8 -*-
"""
This script re-simulates observation data for the
`LoDoPaB-CT dataset <https://doi.org/10.5281/zenodo.3384092>`_.

Prerequisites:

    * imported libraries + ``astra-toolbox==1.8.3`` (requires Python 3.6)
    * unzipped LoDoPaB-CT dataset stored in `PATH`, at least the ground truth
"""
import os
from itertools import islice
from math import ceil
import numpy as np
import odl
from tqdm import tqdm
from skimage.transform import resize
import h5py
import multiprocessing


# path to lodopab dataset (input and output)
PATH = '/localdata/lodopab'
# name for the resimulated observations, the output HDF5 files will be named
# e.g. '{OBSERVATION_NAME}_train_000.hdf5'
OBSERVATION_NAME = 'observation_resimulated'

# mean photon count without attenuation
PHOTONS_PER_PIXEL = 4096

NUM_ANGLES = 1000
NUM_DET_PIXELS = 513

# original ground truth and reconstruction image shape
RECO_IM_SHAPE = (362, 362)
# image shape for simulation
IM_SHAPE = (1000, 1000)  # images will be scaled up from (362, 362)

# ~26cm x 26cm images
MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]

# linear attenuations in m^-1
MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

NUM_SAMPLES_PER_FILE = 128
LEN = {
    'train': 35820,
    'validation': 3522,
    'test': 3553}

IMPL = 'astra_cpu'


reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT,
                               shape=RECO_IM_SHAPE, dtype=np.float32)
space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                          dtype=np.float64)

reco_geometry = odl.tomo.parallel_beam_geometry(
    reco_space, num_angles=NUM_ANGLES, det_shape=(NUM_DET_PIXELS,))
geometry = odl.tomo.parallel_beam_geometry(
    space, num_angles=NUM_ANGLES, det_shape=(NUM_DET_PIXELS,))

reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry, impl=IMPL)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)


def ground_truth_gen(part='train'):
    num_files = ceil(LEN[part] / NUM_SAMPLES_PER_FILE)
    for i in range(num_files):
        with h5py.File(
                os.path.join(PATH, 'ground_truth_{}_{:03d}.hdf5'
                                   .format(part, i)), 'r') as file:
            ground_truth_data = file['data'][:]
        for gt_arr in ground_truth_data:
            yield reco_space.element(gt_arr)


rs = np.random.RandomState(4)


def forward_fun(im):
    # upsample ground_truth from RECO_IM_SHAPE to IM_SHAPE in each dimension
    # before application of forward operator in order to avoid
    # inverse crime
    im_resized = resize(im * MU_MAX, IM_SHAPE, order=1)

    # apply forward operator
    data = ray_trafo(im_resized).asarray()

    data *= (-1)
    np.exp(data, out=data)
    data *= PHOTONS_PER_PIXEL
    return data


for part in ['train', 'validation', 'test']:
    gen = ground_truth_gen(part)
    n_files = ceil(LEN[part] / NUM_SAMPLES_PER_FILE)
    for filenumber in tqdm(range(n_files), desc=part):
        obs_filename = os.path.join(
            PATH,
            '{}_{}_{:03d}.hdf5'.format(OBSERVATION_NAME, part, filenumber))
        with h5py.File(obs_filename, 'w') as observation_file:
            observation_dataset = observation_file.create_dataset(
                'data', shape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                maxshape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                dtype=np.float32, chunks=True)
            im_buf = [im for im in islice(gen, NUM_SAMPLES_PER_FILE)]
            with multiprocessing.Pool(20) as pool:
                data_buf = pool.map(forward_fun, im_buf)

            for i, (im, data) in enumerate(zip(im_buf, data_buf)):
                data = rs.poisson(data) / PHOTONS_PER_PIXEL
                np.maximum(0.1 / PHOTONS_PER_PIXEL, data, out=data)
                np.log(data, out=data)
                data /= (-MU_MAX)
                observation_dataset[i] = data

            # resize last file
            if filenumber == n_files - 1:
                observation_dataset.resize(
                    LEN[part] - (n_files - 1) * NUM_SAMPLES_PER_FILE,
                    axis=0)
