import numpy as np
import tqdm
import datajoint as dj
import scipy
from .sim import Fluorescence, Detection, Tissue

schema = dj.schema('photonics')


@schema
class Sample(dj.Lookup):
    definition = """
    sample : tinyint unsigned 
    ---
    density : int  # cells per cubic mm
    """
    contents = [
        (0, 1000), (1, 3000), (2, 5000), (3, 10_000),
        (4, 20_000), (5, 35_000), (6, 50_000), (7, 75_000), (8, 100_000)]


@schema
class IlluminationCycle(dj.Computed):
    definition = """
    -> Fluorescence
    -> Detection
    ---
    nframes  :  smallint unsigned  # number of illumination frames
    illumination : longblob        # frames x emitters
    """

    def make(self, key):
        emission = np.stack(
            [x for x in (Fluorescence.EPixel & key).fetch('photons_per_cell')])  # emitters x sources
        detection = np.stack(
            [x for x in (Detection.DPixel & key).fetch('detect_probabilities')])  # detectors x sources
        assert emission.dtype == np.float32 and detection.dtype == np.float32
        target_rank = min(
            emission.shape[1],
            detection.shape[0] * emission.shape[0],
            (Tissue & key).fetch1('inner_count')*2)
        illumination = np.identity(emission.shape[0], dtype=np.uint8)
        nframes = int(np.ceil(target_rank / detection.shape[0]))

        qq = emission @ detection.T
        qq = qq @ qq.T

        # combine illumination patterns with minimum overlap
        for _ in tqdm.tqdm(range(illumination.shape[0] - nframes)):
            i, j = np.triu_indices(qq.shape[1], 1)
            ix = np.argmin(qq[i, j])
            i, j = i[ix], j[ix]
            illumination[i] += illumination[j]
            illumination = np.delete(illumination, j, 0)
            qq[i, :] += qq[j, :]
            qq[:, i] += qq[:, j]
            qq = np.delete(qq, j, 0)
            qq = np.delete(qq, j, 1)

        self.insert1(dict(key, nframes=nframes, illumination=illumination))


@schema
class Demix(dj.Computed):
    definition = """
    -> IlluminationCycle
    -> Sample
    ---
    selection : longblob  # selected cells
    mix_norm : longblob    #  cell's mixing vector norm
    demix_norm : longblob  #  cell's demixing vector norm
    bias_norm : longblob  #  cell's bias vector norm
    trans_bias_norm : longblob # don't use. Saved just in case of wrong axis choice
    avg_emitter_power : float  # (uW) when on
    """

    def make(self, key):
        dt = 0.02  # (s) sample duration (one illumination cycle)
        power = 0.04  # Total milliwatts to the brain
        dark_noise = 300  # counts per second
        seed = 0

        # load the emission and detection matrices
        npoints, volume = (Tissue & key).fetch1('npoints', 'volume')
        target_density = (Sample & key).fetch1('density')

        selection = np.r_[:npoints] < int(np.round(target_density) * volume)
        np.random.seed(seed)
        np.random.shuffle(selection)

        illumination = (IlluminationCycle & key).fetch1('illumination')
        nframes = illumination.shape[0]
        illumination = power * illumination / illumination.sum()  # watts averaged over the entire cycle
        avg = nframes * illumination[illumination > 0].mean()

        emission = np.stack(
            [x[selection] for x in (Fluorescence.EPixel & key).fetch('photons_per_cell')])  # emitters x sources
        detection = np.stack(
            [x[selection] for x in (Detection.DPixel & key).fetch('detect_probabilities')])  # detectors x sources

        # construct the mixing matrix mix: nchannels x ncells
        # mix = number of photons from neuron per frame at full fluorescence
        ncells = detection.shape[1]
        ndetectors = detection.shape[0]
        nchannels = nframes * ndetectors
        mix = np.ndarray(dtype='float32', shape=(nchannels, ncells))
        for ichannel in range(0, nchannels, ndetectors):
            mix[ichannel:ichannel + ndetectors] = detection * emission[ichannel // ndetectors]

        # normalize channels by their noise
        mean_fluorescence = 0.03
        nu = dark_noise * dt / nframes
        weights = 1 / np.sqrt(mix.sum(axis=1, keepdims=True) * mean_fluorescence + nu)  # used to be axis=0
        mix *= weights

        # regularized demix matrix
        kmax = 1e6
        square = mix.T @ mix
        identity = np.identity(mix.shape[1])
        alpha = np.sqrt(scipy.linalg.eigh(
            square, eigvals_only=True, eigvals=(ncells - 1, ncells - 1))[0]) / (2 * kmax)
        square += alpha ** 2 * identity
        demix = np.linalg.inv(square) @ mix.T

        # bias matrix
        bias = demix @ mix - identity

        self.insert1(dict(
            key,
            selection=selection,
            avg_emitter_power=avg * 1e6,
            mix_norm=np.linalg.norm(mix, axis=0),
            demix_norm=np.linalg.norm(demix, axis=1),
            bias_norm=np.linalg.norm(bias, axis=1),
            trans_bias_norm=np.linalg.norm(bias, axis=0)))
