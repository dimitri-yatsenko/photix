import numpy as np
import datajoint as dj
from .tracer import SpaceTracer
from matplotlib import pyplot as plt

schema = dj.schema('photixxx')


@schema
class DSim(dj.Lookup):
    definition = """
    # Detector Field Specification
    dsim : int
    --- 
    detector_type='one-sided' : varchar(30)   # choice in simulation
    detector_width=10.00: decimal(5,2)   # (um) along x-axis
    detector_height=10.00:  decimal(5,2)  # (um) along y-axis
    anisotropy = 0.88 : float  # factor in the Henyey-Greenstein formula
    absorption_length: float # (um)  average travel path before a absoprtion event 
    scatter_length : float # (um) average travel path before a scatter event
    volume_dimx = 1000 : int unsigned # (voxels)
    volume_dimy = 1000 : int unsigned # (voxels)
    volume_dimz = 1000 : int unsigned # (voxels)
    pitch = 2.2 : float  # (um)  spatial sampling period of the model volume
    """

    contents = [
        dict(dsim=0, detector_type='one-sided', detector_height=50, scatter_length=50, absorption_length=1.5e4),
        dict(dsim=1, detector_type='one-sided', detector_height=20, scatter_length=50, absorption_length=1.5e4),
        dict(dsim=2, detector_type='narrowed2', detector_height=20, scatter_length=50, absorption_length=1.5e4),
        dict(dsim=4, detector_type='narrowed4', detector_height=20, scatter_length=50, absorption_length=1.5e4),
        dict(dsim=8, detector_type='narrowed8', detector_height=20, scatter_length=50, absorption_length=1.5e4),
        dict(dsim=10, detector_type='narrowed10', detector_height=20, scatter_length=50, absorption_length=1.5e4),
        dict(dsim=14, detector_type='narrowed4', detector_height=20, scatter_length=50, absorption_length=1500),
        dict(dsim=18, detector_type='narrowed8', detector_height=20, scatter_length=50, absorption_length=500)]

    def make_volume(self, hops=100_000):
        spec = self.fetch1()
        kwargs = {k: spec[k] for k in spec if k in {
            'pitch', 'anisotropy', 'scatter_length', 'absorption_length', 'detector_type'}}

        kwargs.update(
            dims=tuple(spec[k] for k in ('volume_dimx', 'volume_dimy', 'volume_dimz')),
            emitter_spread='spherical',
            emitter_size=(float(spec['detector_width']), float(spec['detector_height']), 0))

        space = SpaceTracer(**kwargs)
        space.run(hops=hops)
        space.volume *= space.emitter_area
        return space


@schema
class ESim(dj.Lookup):
    definition = """
    # Emission Field Specification
    esim : int
    --- 
    beam_compression : float  
    y_steer : float   # the steer angle in the plane of the shank
    emitter_width=10.00: decimal(5,2)   # (um) along x-axis
    emitter_height=10.00:  decimal(5,2)  # (um) along y-axis
    anisotropy = 0.88 : float  # factor in the Henyey-Greenstein formula
    absorption_length: float # (um)  average travel path before a absoprtion event    
    scatter_length: float # (um) average travel path before a scatter event
    volume_dimx = 1000 : int unsigned # (voxels)
    volume_dimy = 1000 : int unsigned # (voxels)
    volume_dimz = 1000 : int unsigned # (voxels)
    beam_xy_aspect = 1.0 : float   # compression of y. E.g. 2.0 means that y is compressed by factor of 2
    pitch = 2.2 : float  # (um)  spatial sampling period of the model volume
    """

    contents = [
        dict(esim=0, beam_compression=1.0, y_steer=0.0, scatter_length=50, absorption_length=1.5e4),

        dict(esim=10, beam_compression=1 / 4, y_steer=-np.pi / 3, scatter_length=50, absorption_length=1.5e4),
        dict(esim=11, beam_compression=1 / 4, y_steer=-np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=12, beam_compression=1 / 4, y_steer=-np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=13, beam_compression=1 / 4, y_steer=-np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=14, beam_compression=1 / 4, y_steer=0,         scatter_length=50, absorption_length=1.5e4),
        dict(esim=15, beam_compression=1 / 4, y_steer=+np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=16, beam_compression=1 / 4, y_steer=+np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=17, beam_compression=1 / 4, y_steer=+np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=18, beam_compression=1 / 4, y_steer=+np.pi / 3, scatter_length=50, absorption_length=1.5e4),

        dict(esim=20, beam_compression=1 / 3, y_steer=-np.pi / 3, scatter_length=50, absorption_length=1.5e4),
        dict(esim=21, beam_compression=1 / 3, y_steer=-np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=22, beam_compression=1 / 3, y_steer=-np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=23, beam_compression=1 / 3, y_steer=-np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=24, beam_compression=1 / 3, y_steer=0,           scatter_length=50, absorption_length=1.5e4),
        dict(esim=25, beam_compression=1 / 3, y_steer=+np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=26, beam_compression=1 / 3, y_steer=+np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=27, beam_compression=1 / 3, y_steer=+np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=28, beam_compression=1 / 3, y_steer=+np.pi / 3, scatter_length=50, absorption_length=1.5e4),

        dict(esim=30, beam_compression=1 / 6, y_steer=-np.pi / 3, scatter_length=50, absorption_length=1.5e4),
        dict(esim=31, beam_compression=1 / 6, y_steer=-np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=32, beam_compression=1 / 6, y_steer=-np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=33, beam_compression=1 / 6, y_steer=-np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=34, beam_compression=1 / 6, y_steer=0, scatter_length=50, absorption_length=1.5e4),
        dict(esim=35, beam_compression=1 / 6, y_steer=+np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=36, beam_compression=1 / 6, y_steer=+np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=37, beam_compression=1 / 6, y_steer=+np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=38, beam_compression=1 / 6, y_steer=+np.pi / 3, scatter_length=50, absorption_length=1.5e4),

        dict(esim=40, beam_compression=1 / 12, y_steer=-np.pi / 3, scatter_length=50, absorption_length=1.5e4),
        dict(esim=41, beam_compression=1 / 12, y_steer=-np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=42, beam_compression=1 / 12, y_steer=-np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=43, beam_compression=1 / 12, y_steer=-np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=44, beam_compression=1 / 12, y_steer=0, scatter_length=50, absorption_length=1.5e4),
        dict(esim=45, beam_compression=1 / 12, y_steer=+np.pi / 12, scatter_length=50, absorption_length=1.5e4),
        dict(esim=46, beam_compression=1 / 12, y_steer=+np.pi / 6, scatter_length=50, absorption_length=1.5e4),
        dict(esim=47, beam_compression=1 / 12, y_steer=+np.pi / 4, scatter_length=50, absorption_length=1.5e4),
        dict(esim=48, beam_compression=1 / 12, y_steer=+np.pi / 3, scatter_length=50, absorption_length=1.5e4),

        # tissue tanning
        dict(esim=130, beam_compression=1 / 6, y_steer=-np.pi / 3, scatter_length=50, absorption_length=500),
        dict(esim=131, beam_compression=1 / 6, y_steer=-np.pi / 4, scatter_length=50, absorption_length=500),
        dict(esim=132, beam_compression=1 / 6, y_steer=-np.pi / 6, scatter_length=50, absorption_length=500),
        dict(esim=133, beam_compression=1 / 6, y_steer=-np.pi / 12, scatter_length=50, absorption_length=500),
        dict(esim=134, beam_compression=1 / 6, y_steer=0, scatter_length=50, absorption_length=500),
        dict(esim=135, beam_compression=1 / 6, y_steer=+np.pi / 12, scatter_length=50, absorption_length=500),
        dict(esim=136, beam_compression=1 / 6, y_steer=+np.pi / 6, scatter_length=50, absorption_length=500),
        dict(esim=137, beam_compression=1 / 6, y_steer=+np.pi / 4, scatter_length=50, absorption_length=500),
        dict(esim=138, beam_compression=1 / 6, y_steer=+np.pi / 3, scatter_length=50, absorption_length=500),

        dict(esim=140, beam_compression=1 / 12, y_steer=-np.pi / 3, scatter_length=50, absorption_length=500),
        dict(esim=141, beam_compression=1 / 12, y_steer=-np.pi / 4, scatter_length=50, absorption_length=500),
        dict(esim=142, beam_compression=1 / 12, y_steer=-np.pi / 6, scatter_length=50, absorption_length=500),
        dict(esim=143, beam_compression=1 / 12, y_steer=-np.pi / 12, scatter_length=50, absorption_length=500),
        dict(esim=144, beam_compression=1 / 12, y_steer=0, scatter_length=50, absorption_length=500),
        dict(esim=145, beam_compression=1 / 12, y_steer=+np.pi / 12, scatter_length=50, absorption_length=500),
        dict(esim=146, beam_compression=1 / 12, y_steer=+np.pi / 6, scatter_length=50, absorption_length=500),
        dict(esim=147, beam_compression=1 / 12, y_steer=+np.pi / 4, scatter_length=50, absorption_length=500),
        dict(esim=148, beam_compression=1 / 12, y_steer=+np.pi / 3, scatter_length=50, absorption_length=500),


        dict(esim=320, beam_compression=1 / 3, y_steer=-np.pi / 3, scatter_length=50, absorption_length=1500),
        dict(esim=321, beam_compression=1 / 3, y_steer=-np.pi / 4, scatter_length=50, absorption_length=1500),
        dict(esim=322, beam_compression=1 / 3, y_steer=-np.pi / 6, scatter_length=50, absorption_length=1500),
        dict(esim=323, beam_compression=1 / 3, y_steer=-np.pi / 12, scatter_length=50, absorption_length=1500),
        dict(esim=324, beam_compression=1 / 3, y_steer=0, scatter_length=50, absorption_length=1500),
        dict(esim=325, beam_compression=1 / 3, y_steer=+np.pi / 12, scatter_length=50, absorption_length=1500),
        dict(esim=326, beam_compression=1 / 3, y_steer=+np.pi / 6, scatter_length=50, absorption_length=1500),
        dict(esim=327, beam_compression=1 / 3, y_steer=+np.pi / 4, scatter_length=50, absorption_length=1500),
        dict(esim=328, beam_compression=1 / 3, y_steer=+np.pi / 3, scatter_length=50, absorption_length=1500),

        # tissue tanning
        dict(esim=330, beam_compression=1 / 6, y_steer=-np.pi / 3, scatter_length=50, absorption_length=1500),
        dict(esim=331, beam_compression=1 / 6, y_steer=-np.pi / 4, scatter_length=50, absorption_length=1500),
        dict(esim=332, beam_compression=1 / 6, y_steer=-np.pi / 6, scatter_length=50, absorption_length=1500),
        dict(esim=333, beam_compression=1 / 6, y_steer=-np.pi / 12, scatter_length=50, absorption_length=1500),
        dict(esim=334, beam_compression=1 / 6, y_steer=0, scatter_length=50, absorption_length=1500),
        dict(esim=335, beam_compression=1 / 6, y_steer=+np.pi / 12, scatter_length=50, absorption_length=1500),
        dict(esim=336, beam_compression=1 / 6, y_steer=+np.pi / 6, scatter_length=50, absorption_length=1500),
        dict(esim=337, beam_compression=1 / 6, y_steer=+np.pi / 4, scatter_length=50, absorption_length=1500),
        dict(esim=338, beam_compression=1 / 6, y_steer=+np.pi / 3, scatter_length=50, absorption_length=1500),]

    def make_volume(self, hops=100_000):
        spec = self.fetch1()

        kwargs = {k: spec[k] for k in spec if k in {
            'pitch', 'anisotropy', 'scatter_length',
            'y_steer', 'beam_compression', 'beam_xy_aspect',
            'absorption_length'}}

        kwargs.update(
            dims=tuple(spec[k] for k in ('volume_dimx', 'volume_dimy', 'volume_dimz')),
            emitter_size=(float(spec['emitter_width']), float(spec['emitter_height']), 0))

        space = SpaceTracer(**kwargs)
        space.run(hops=hops)
        return space

