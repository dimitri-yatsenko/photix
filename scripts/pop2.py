from photix import sim

r = 'design in (101, 102, 104, 105, 106, 108, 115, 111, 215, 211, 304, 314, 306, 316)'
sim.Tissue.populate(r, display_progress=True, reserve_jobs=True)
sim.Detection.populate(r, display_progress=True, reserve_jobs=True)
sim.Fluorescence.populate(r, display_progress=True, reserve_jobs=True)
