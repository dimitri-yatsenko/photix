import datajoint as dj
import numpy as np
from scipy import spatial
from . import design
from . tracer import ETracer

schema = dj.schema('photix_sim')


@schema
class Tissue(dj.Computed):
    definition = """
    -> design.Geometry
    ---
    density : float #  points per mm^3
    margin : float # (um) margin to include on boundaries
    min_distance : float  # (um)
    points  : longblob  # cell xyz
    npoints : int    # total number of points in volume
    inner_count : int  # number of points inside the probe boundaries 
    volume : float  # (mm^3), hull volume including outer points
    """

    def make(self, key):
        density = 110000  # per cubic mm
        xyz = np.stack((design.Geometry.EPixel() & key).fetch('e_loc'))
        margin = 150
        bounds_min = xyz.min(axis=0) - margin
        bounds_max = xyz.max(axis=0) + margin
        volume = (bounds_max - bounds_min).prod() * 1e-9
        npoints = int(volume * density + 0.5)

        # generate random points that aren't too close
        min_distance = 10.0  # cells aren't not allowed any closer
        points = np.empty((npoints, 3), dtype='float32')
        replace = np.r_[:npoints]
        while replace.size:
            points[replace, :] = np.random.rand(replace.size, 3) * (bounds_max - bounds_min) + bounds_min
            replace = spatial.cKDTree(points).query_pairs(min_distance, output_type='ndarray')[:, 0]

        # eliminate points that are too distant
        inner = (spatial.Delaunay(xyz).find_simplex(points)) != -1
        d, _ = spatial.cKDTree(points[inner, :]).query(points[~inner, :], distance_upper_bound=margin)
        points = np.vstack((points[inner, :], points[~inner, :][d < margin, :]))

        self.insert1(dict(
            key, margin=margin,
            density=density,
            npoints=points.shape[0], min_distance=min_distance,
            points=points,
            volume=spatial.ConvexHull(points).volume * 1e-9,
            inner_count=inner.sum()))


@schema
class Detection(dj.Computed):
    definition = """
    -> Tissue
    -> design.Geometry.DPixel
    ---
    detection  : longblob  # probability per cell
    photon_count  :  int unsigned
    """


@schema
class Emission(dj.Computed):
    definition = """
    -> Tissue
    -> design.Geometry.EPixel
    ---
    flux  : longblob  # dims = points x steers probability flux per um^2
    photon_count  :  int unsigned
    """

    def make(self, key):
        # rotate points into the EPixel's coordinate system: z-axis = normal
        config = (design.Design & key).fetch1()

        loc, norm = (design.Geometry.EPixel & key).fetch1('e_loc', 'e_norm')
        z_basis = np.pad(norm, (0, 1))
        y_basis = np.array([0, 0, 1])
        basis = np.stack((np.cross(z_basis, y_basis), y_basis, z_basis))
        assert np.allclose(basis @ basis.T, np.eye(3))
        points = ((Tissue & key).fetch1('points') - loc.astype('float32')) @ basis

        steer = 0.0
        beam_compression = 1.0
        beam_xy_aspect = 1.0

        tracer = ETracer(
            emitter_shape="rect",
            emitter_size=(float(config['epixel_width']), float(config['epixel_width']), 0),
            emitter_profile=config['epixel_profile'],
            detector_positions=points,
            anisotropy=config['anisotropy'],
            absorption_length=config['absorption_length'],
            scatter_length=config['scatter_length'],
            y_steer=steer,
            beam_compression=beam_compression,
            beam_xy_aspect=beam_xy_aspect,
            max_hop=20.0)

        tracer.run(500, display_progress=False)
        self.insert1(dict(key, flux=tracer.flux, photon_count=tracer.photon_count))