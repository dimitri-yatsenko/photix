import datajoint as dj
import numpy as np
from scipy import spatial
from . import design
from . tracer import ETracer

schema = dj.schema('photix')


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
