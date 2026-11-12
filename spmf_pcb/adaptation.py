"""Apply the paper's learned reliability and bounded modulation settings."""

def configure(module, settings):
    if not module.reliability_on or not module.use_gate:
        raise ValueError("SPMF requires its reliability branch and gates.")
    module.force_r = float(settings["reliability_floor"])
    module.force_steps = int(settings["reliability_floor_steps"])
    module.mod_cap = float(settings["modulation_cap"])
    check(module, settings)

def check(module, settings):
    expected = (settings["reliability_floor"], settings["reliability_floor_steps"], settings["modulation_cap"])
    actual = (module.force_r, module.force_steps, module.mod_cap)
    if actual != expected or not module.reliability_on or not module.use_gate:
        raise ValueError("Checkpoint does not use the requested SPMF settings: " + str(actual))
