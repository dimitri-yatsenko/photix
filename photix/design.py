import numpy as np
import datajoint as dj
from . import fields
import itertools
import json
import tqdm

schema = dj.schema('photixxx')


@schema
class Design(dj.Lookup):
    definition = """
    design : int 
    ---
    lattice : varchar(16)  # lattice type: sqdiag, hex
    lattice_rows : tinyint  
    lattice_pitch : float # um
    epixel_depths: varchar(200)   # (degrees) start:stop:step, group
    epixel_azimuths: varchar(200)   # (degrees) start:stop:step  
    dpixel_depths: varchar(200)   # (degrees) start:stop:step, group
    dpixel_azimuths: varchar(200)   # (degrees) start:stop:step
    field_sims : varchar(200)  # json string specifying dfields and efields
    """
    contents = [
        dict(
            design=101,
            lattice='sqdiag',
            lattice_rows=7,
            lattice_pitch=200,
            epixel_depths="0:450:50,8",
            epixel_azimuths="0:3240:45",
            dpixel_depths="25:420:50,2",
            dpixel_azimuths="0:2880:180",
            field_sims='{"d": 0, "e": [0]}'),
        dict(
            design=102,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 1, "e": [10, 11, 12, 13, 14, 15, 16, 17, 18]}'),
        dict(
            design=104,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 4, "e": [20, 21, 22, 23, 24, 25, 26, 27, 28]}'),
        dict(
            design=105,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 8, "e": [20, 21, 22, 23, 24, 25, 26, 27, 28]}'),
        dict(
            design=106,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 8, "e": [30, 31, 32, 33, 34, 35, 36, 37, 38]}'),
        dict(
            design=107,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 10, "e": [30, 31, 32, 33, 34, 35, 36, 37, 38]}'),
        dict(
            design=108,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=150,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 10, "e": [40, 41, 42, 43, 44, 45, 46, 47, 48]}'),
        dict(
            design=109,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=120,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 10, "e": [40, 41, 42, 43, 44, 45, 46, 47, 48]}'),
        dict(
            design=110,
            lattice='hex',
            lattice_rows=7,
            lattice_pitch=120,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 10, "e": [40, 41, 42, 43, 44, 45, 46, 47, 48]}'),
        dict(
            design=111,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=150,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 8, "e": [40, 41, 42, 43, 44, 45, 46, 47, 48]}'),
        dict(
            design=115,
            design_description="",
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 8, "e": [40, 41, 42, 43, 44, 45, 46, 47, 48]}'),
        dict(
            design=211,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=150,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 18, "e": [140, 141, 142, 143, 144, 145, 146, 147, 148]}'),
        dict(
            design=215,
            design_description="",
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 18, "e": [140, 141, 142, 143, 144, 145, 146, 147, 148]}'),

        dict(
            design=304,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 14, "e": [320, 321, 322, 323, 324, 325, 326, 327, 328]}'),

        dict(
            design=314,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=150,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 14, "e": [320, 321, 322, 323, 324, 325, 326, 327, 328]}'),

        dict(
            design=306,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 14, "e": [330, 331, 332, 333, 334, 335, 336, 337, 338]}'),

        dict(
            design=316,
            lattice='hex',
            lattice_rows=5,
            lattice_pitch=150,
            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            field_sims='{"d": 14, "e": [330, 331, 332, 333, 334, 335, 336, 337, 338]}'),

    ]

    @staticmethod
    def make_lattice(lattice_type, nrows):
        if lattice_type == 'hex':
            return np.array([
                (x, y * 3 ** .5 / 2)
                for y in np.arange(-(nrows - 1) / 2, nrows / 2)
                for x in np.arange(-(nrows - abs(y) - 1) / 2, (nrows - abs(y)) / 2)])
        if lattice_type == 'sqdiag':
            return np.array([
                (x, y - (nrows - 1) / 2)
                for y in range(nrows)
                for x in np.arange(-(nrows - 1) / 2 + (1 - y % 2), nrows / 2, 2)])
        raise ValueError('Invalid lattice_type')


@schema
class Geometry(dj.Computed):
    definition = """
    -> Design
    ---
    shanks_xy  : longblob  # n x 2    
    n_shanks  : smallint unsigned
    """

    class EPixel(dj.Part):
        definition = """
        -> master
        epixel : smallint unsigned
        ---
        e_loc : longblob   # x, y, z  location of center
        e_norm : longblob   # x, y  unit vector 
        """

    class DPixel(dj.Part):
        definition = """
        -> master
        dpixel : smallint unsigned
        ---
        d_loc : longblob   # x, y, z  location of center
        d_norm : longblob   # x, y  unit vector 
        """

    class EField(dj.Part):
        definition = """
        -> master.EPixel
        -> fields.ESim
        """

    class DField(dj.Part):
        definition = """
        -> master.DPixel
        ---
        -> fields.DSim
        """

    def make(self, key):
        design = (Design & key).fetch1()

        shanks_xy = Design.make_lattice(
            design['lattice'], design['lattice_rows']) * design['lattice_pitch']
        field_sims = json.loads(design['field_sims'])
        esims = field_sims['e']
        dsim = field_sims['d']
        assert {'dsim': dsim} in fields.DSim(), f"DSim {dsim} not found"
        try:
            err = next(esim for esim in esims if {'esim': esim} not in fields.ESim())
            raise ValueError(err)
        except StopIteration:
            pass
        self.insert1(dict(key,
                          shanks_xy=np.array(shanks_xy),
                          n_shanks=len(shanks_xy)))
        ecount = itertools.count()
        dcount = itertools.count()

        for xy in tqdm.tqdm(shanks_xy):
            # EPixels
            azimuths = np.arange(*[float(x)
                                   for x in design['epixel_azimuths'].split(':')])
            depths, group = design['epixel_depths'].split(',')
            depths = np.arange(*[float(x) for x in depths.split(':')])
            pos = np.vstack([np.array([[xy[0], xy[1], d]] * int(group)) for d in depths])
            norm = np.stack((np.cos(azimuths / 180 * np.pi), np.sin(azimuths / 180 * np.pi))).T
            assert pos.shape[0] == norm.shape[0], "Invalid emitter positions specification"
            for c, p, n in zip(ecount, pos, norm):
                self.EPixel().insert1(dict(key, epixel=c, e_loc=p, e_norm=n))
                self.EField().insert(dict(key, epixel=c, esim=esim) for esim in esims)

            # D pixels
            azimuths = np.arange(*[float(x)
                                   for x in design['dpixel_azimuths'].split(':')])
            if key['design'] == 101:
                # sorry for special case
                azimuths += xy[1]//200 * 90
            depths, group = design['dpixel_depths'].split(',')
            depths = np.arange(*[float(x) for x in depths.split(':')])
            pos = np.vstack([np.array([[xy[0], xy[1], d]] * int(group)) for d in depths])
            norm = np.stack((np.cos(azimuths / 180 * np.pi), np.sin(azimuths / 180 * np.pi))).T
            assert pos.shape[0] == norm.shape[0], "Invalid emitter positions specification"
            for c, p, n in zip(dcount, pos, norm):
                self.DPixel().insert1(dict(key, dpixel=c, d_loc=p, d_norm=n))
                self.DField().insert1(dict(key, dpixel=c, dsim=dsim))
