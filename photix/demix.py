import numpy as np
import tqdm
import datajoint as dj
import scipy
from .sim import Fluorescence, Detection, Tissue

schema = dj.schema('photixxx')


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
            [x for x in (Fluorescence.EField & key).fetch('emit_probabilities')])  # emitters x sources
        detection = np.stack(
            [x for x in (Detection.DField & key).fetch('detect_probabilities')])  # detectors x sources
        assert emission.dtype == np.float32 and detection.dtype == np.float32
        npoints, density = (Tissue & key).fetch1('npoints', 'density')
        target_rank = npoints / density * 120000
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
    dt  : float   # s
    dark_noise :  float  # s^-1
    emitter_power : float  # mW
    mean_fluorescence : float  # fraction
    selection : longblob  # selected cells
    mix_norm : longblob    #  cell's mixing vector norm
    demix_norm : longblob  #  cell's demixing vector norm
    bias_norm : longblob  #  cell's bias vector norm
    trans_bias_norm : longblob # don't use. Saved just in case of wrong axis choice
    total_power : float  # mW
    """

    def make(self, key):
        dt = 0.002  # (s) sample duration (one illumination cycle)
        dark_noise = 300  # counts per second
        seed = 0
        emitter_power = 1e-4  # 100 uW
        detector_efficiency = 0.6
        mean_fluorescence = 0.05  # e.g. 0.03 = 0.05 times 60% detector efficiency
        photons_per_joule = 2.4e18

        # load the emission and detection matrices
        npoints, volume = (Tissue & key).fetch1('npoints', 'volume')
        target_density = (Sample & key).fetch1('density')

        selection = np.r_[:npoints] < int(np.round(target_density) * volume)
        np.random.seed(seed)
        np.random.shuffle(selection)

        illumination = (IlluminationCycle & key).fetch1('illumination')
        nframes = illumination.shape[0]
        illumination = emitter_power * illumination * dt / nframes  # joules
        emission = photons_per_joule * mean_fluorescence * np.stack(
            [x[selection] for x in (Fluorescence.EField & key).fetch('emit_probabilities')])  # E-pixels x sources
        emission = illumination @ emission  # photons per frame

        detection = detector_efficiency * np.stack(
            [x[selection] for x in (Detection.DField & key).fetch('detect_probabilities')])  # D-pixels x sources

        # construct the mixing matrix mix: nchannels x ncells
        # mix = number of photons from neuron per frame at full fluorescence
        ncells = detection.shape[1]
        ndetectors = detection.shape[0]
        nchannels = nframes * ndetectors
        mix = np.ndarray(dtype='float32', shape=(nchannels, ncells))
        for ichannel in range(0, nchannels, ndetectors):
            mix[ichannel:ichannel + ndetectors] = detection * emission[ichannel // ndetectors]

        # normalize channels by their noise
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
            dt=dt,
            dark_noise=dark_noise,
            emitter_power=emitter_power*1e3,
            mean_fluorescence=mean_fluorescence,
            selection=selection,
            total_power=illumination.sum()/dt*1e3,
            mix_norm=np.linalg.norm(mix, axis=0),
            demix_norm=np.linalg.norm(demix, axis=1),
            bias_norm=np.linalg.norm(bias, axis=1),
            trans_bias_norm=np.linalg.norm(bias, axis=0)))


@schema
class Cosine(dj.Computed):
    definition = """
    -> Demix
    ---
    cosines : longblob
    """

    def make(self, key):
        max_bias = 0.01
        mix_norm, demix_norm, bias_norm = (Demix & key).fetch1('mix_norm', 'demix_norm', 'bias_norm')
        cosines = (bias_norm < max_bias) / (mix_norm * demix_norm)
        self.insert1(dict(key, cosines=cosines))


@schema
class SpikeSNR(dj.Computed):
    definition = """
    -> Demix
    ---
    snr : longblob
    tau : float
    delta : float
    rho : float
    avg_snr : float
    frac_above_1 : float
    """

    def make(self, key):
        max_bias = 0.01
        tau = 1.5
        dt, mean_fluorescence, inner_count, selection, demix_norm, bias = (
                Demix * Tissue & key).fetch1(
            'dt', 'mean_fluorescence',
            'inner_count', 'selection', 'demix_norm', 'bias_norm')
        inner = selection.copy()
        inner[inner_count:] = False  # exclude cells outside the probe
        inner = inner[selection]
        delta = mean_fluorescence * 0.4
        demix_norm, bias = (Demix & key).fetch1('demix_norm', 'bias_norm')
        h = np.exp(-np.r_[0:6 * tau:dt] / tau)
        rho = np.sqrt((h**2).sum())/h[0]
        snr = (bias < max_bias) * rho * delta / demix_norm

        self.insert1(dict(key,
                          snr=snr, delta=delta, rho=rho, tau=tau, avg_snr=snr[inner].mean(),
                          frac_above_1=(snr[inner] >= 1.0).mean()))
