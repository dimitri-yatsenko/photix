import numpy as np
import tqdm
import datajoint as dj
import scipy
from .sim import Fluorescence, Detection, Tissue

schema = dj.schema('photix')


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
