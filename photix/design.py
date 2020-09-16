import numpy as np
import datajoint as dj
import itertools

schema = dj.schema('photix')


@schema
class Design(dj.Lookup):
    definition = """
    design : int 
    ---
    design_description : varchar(300)     

    lattice : varchar(16)  # lattice type: sqdiag, hex
    lattice_rows : tinyint  
    lattice_pitch : float # um

    epixel_depths: varchar(200)   # (degrees) start:stop:step, group
    epixel_azimuths: varchar(200)   # (degrees) start:stop:step  
    epixel_width: decimal(4,2)
    epixel_height: decimal(4,2)
    epixel_profile: varchar(300)   # emission field spec
    epixel_steer_angles: varchar(300)  # (degrees) start:stop:step 

    dpixel_depths: varchar(200)   # (degrees) start:stop:step, group
    dpixel_azimuths: varchar(200)   # (degrees) start:stop:step
    dpixel_width: decimal(4,2)   # (um) along x-axis
    dpixel_height:  decimal(4,2)  # (um) along y-axis
    dpixel_profile: varchar(300)  # detection field spec

    anisotropy: float  # factor in the Henyey-Greenstein formula
    absorption_length: float # (um)  average travel path before a absoprtion event 
    scatter_length: float # (um) average travel path before a scatter event

    detector_efficiency : float 
    """

    contents = [

        dict(
            design=1,
            design_description="Shepherd/Roukes original",

            lattice='sqdiag',
            lattice_rows=7,
            lattice_pitch=200,

            epixel_depths="0:450:50,8",
            epixel_azimuths="0:3240:45",
            epixel_width=10,
            epixel_height=10,
            epixel_profile="lambertian",
            epixel_steer_angles="0",

            dpixel_depths="25:420:50,2",
            dpixel_azimuths="0:1440:90",
            dpixel_width=10,
            dpixel_height=50,
            dpixel_profile="lambertian",

            anisotropy=.88,
            absorption_length=14000,
            scatter_length=100,
            detector_efficiency=0.65,
        ),
        dict(
            design=2,
            design_description="Hex-19 pitch 200",

            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,

            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            epixel_width=10,
            epixel_height=10,
            epixel_profile="lambertian",
            epixel_steer_angles="0",

            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            dpixel_width=10,
            dpixel_height=20,
            dpixel_profile="lambertian",

            anisotropy=.88,
            absorption_length=14000,
            scatter_length=100,
            detector_efficiency=0.65,
        ),
        dict(
            design=3,
            design_description="Hex-19 pitch 200 E-Pixels steered & narrowed",

            lattice='hex',
            lattice_rows=5,
            lattice_pitch=200,

            epixel_depths="0:1001:30,1",
            epixel_azimuths="22.5:4600:135",
            epixel_width=10,
            epixel_height=10,
            epixel_profile="lambertian",
            epixel_steer_angles="0",

            dpixel_depths="15:1001:30,1",
            dpixel_azimuths="270:4600:135",
            dpixel_width=10,
            dpixel_height=20,
            dpixel_profile="lambertian",

            anisotropy=.88,
            absorption_length=14000,
            scatter_length=100,
            detector_efficiency=0.65,
        )
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

    def make(self, key):
        design = (Design & key).fetch1()

        shanks_xy = Design.make_lattice(
            design['lattice'], design['lattice_rows']) * design['lattice_pitch']

        self.insert1(dict(key,
                          shanks_xy=np.array(shanks_xy),
                          n_shanks=len(shanks_xy)))
        ecount = itertools.count()
        dcount = itertools.count()
        for xy in shanks_xy:
            # EPixels
            azimuths = np.arange(*[float(x)
                                   for x in design['epixel_azimuths'].split(':')])
            depths, group = design['epixel_depths'].split(',')
            depths = np.arange(*[float(x) for x in depths.split(':')])
            pos = np.vstack([np.array([[xy[0], xy[1], d]] * int(group)) for d in depths])
            norm = np.stack((np.cos(azimuths / 180 * np.pi), np.sin(azimuths / 180 * np.pi))).T
            assert pos.shape[0] == norm.shape[0], "Invalid emitter positions specification"
            self.EPixel().insert(
                dict(key, epixel=c, e_loc=p, e_norm=n)
                for c, p, n in zip(ecount, pos, norm))
            # D pixels
            azimuths = np.arange(*[float(x)
                                   for x in design['dpixel_azimuths'].split(':')])
            depths, group = design['dpixel_depths'].split(',')
            depths = np.arange(*[float(x) for x in depths.split(':')])
            pos = np.vstack([np.array([[xy[0], xy[1], d]] * int(group)) for d in depths])
            norm = np.stack((np.cos(azimuths / 180 * np.pi), np.sin(azimuths / 180 * np.pi))).T
            assert pos.shape[0] == norm.shape[0], "Invalid emitter positions specification"
            self.DPixel().insert(
                dict(key, dpixel=c, d_loc=p, d_norm=n)
                for c, p, n in zip(dcount, pos, norm))