from photix import sim

sim.Tissue.populate(display_progress=True, )
sim.Emission.populate({'design':1}, display_progress=True, reserve_jobs=True)
