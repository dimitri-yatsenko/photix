from scipy import spatial
from . import design
from .design import *
from .fields import *

schema = dj.schema('photixxx')


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
        margin = 75
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
class Fluorescence(dj.Computed):
    definition = """
    -> Tissue
    """

    class EField(dj.Part):
        definition = """
        # Fluorescence produced by cells per Joule of illumination
        -> master
        -> Geometry.EField
        ---
        nphotons : int   # number of simulated photons for the volume
        emit_probabilities  : longblob   # photons emitted from cells per joule of illumination
        mean_probability : float  # mean probability per cell
        """

    def make(self, key):
        neuron_cross_section = 0.1  # um^2
        points = (Tissue & key).fetch1('points')
        self.insert1(key)
        for esim_key in (ESim() & (Geometry.EField & key)).fetch("KEY"):
            pitch, *dims = (ESim & esim_key).fetch1(
                'pitch', 'volume_dimx', 'volume_dimy', 'volume_dimz')
            dims = np.array(dims)
            space = (ESim & esim_key).make_volume(hops=100_000)
            for k in tqdm.tqdm((Geometry.EField & key & esim_key).fetch('KEY')):
                # cell positions in volume coordinates
                e_xyz, basis_z = (Geometry.EPixel & k).fetch1('e_loc', 'e_norm')
                basis_y = np.array([0, 0, 1])
                basis_z = np.append(basis_z, 0)
                basis = np.stack((np.cross(basis_y, basis_z), basis_y, basis_z)).T
                assert np.allclose(basis.T @ basis, np.eye(3)), "incorrect epixel orientation"
                vxyz = np.int16(np.round((points - e_xyz) @ basis / pitch + dims / 2))
                # probabilities
                v = neuron_cross_section * np.array([
                    space.volume[q[0], q[1], q[2]] if
                    0 <= q[0] < dims[0] and
                    0 <= q[1] < dims[1] and
                    0 <= q[2] < dims[2] else 0 for q in vxyz])
                self.EField().insert1(
                    dict(k, **esim_key,
                         nphotons=space.total_count,
                         emit_probabilities=np.float32(v),
                         mean_probability=v.mean()))


@schema
class Detection(dj.Computed):
    definition = """
    -> Tissue
    """

    class DField(dj.Part):
        definition = """
        # Fraction of photons detected from each cell per detector
        -> master
        -> Geometry.DField
        ---
        nphotons : int   # number of simulated photons for the volume
        detect_probabilities  : longblob   # fraction of photons detected from each neuron
        mean_probability : float  # mean probability of detection across all neurons
        """

    def make(self, key):
        points = (Tissue & key).fetch1('points')
        self.insert1(key)
        for dsim_key in (DSim & (Geometry.DField & key)).fetch("KEY"):
            pitch, *dims = (DSim & dsim_key).fetch1(
                'pitch', 'volume_dimx', 'volume_dimy', 'volume_dimz')
            space = (DSim & dsim_key).make_volume(hops=100_000)
            dims = np.array(dims)
            for k in tqdm.tqdm((Geometry.DField & key & dsim_key).fetch('KEY')):
                # cell positions in volume coordinates
                d_xyz, basis_z = (Geometry.DPixel & k).fetch1('d_loc', 'd_norm')
                basis_y = np.array([0, 0, 1])
                basis_z = np.append(basis_z, 0)
                basis = np.stack((np.cross(basis_y, basis_z), basis_y, basis_z)).T
                assert np.allclose(basis.T @ basis, np.eye(3)), "incorrect dpixel orientation"
                vxyz = np.int16(np.round((points - d_xyz) @ basis / pitch + dims / 2))
                # sample DSim volume
                v = np.array([
                    space.volume[q[0], q[1], q[2]] if
                    0 <= q[0] < dims[0] and
                    0 <= q[1] < dims[1] and
                    0 <= q[2] < dims[2] else 0 for q in vxyz])
                self.DField().insert1(
                    dict(k, nphotons=space.total_count,
                         detect_probabilities=np.float32(v),
                         mean_probability=v.mean()))
