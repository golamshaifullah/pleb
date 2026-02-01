"""Shared plot style mappings for QC visualization."""

EVENT_STYLES = {
    "event_member": ("o", "red"),
    "transient": ("o", "red"),
    "exp_dip": ("v", "black"),
    "glitch": ("*", "magenta"),
    "gaussian_bump": ("H", "teal"),
    "solar_event": ("^", "orange"),
    "eclipse_event": ("h", "blue"),
    "step": ("s", "purple"),
    "dm_step": ("D", "green"),
    "step_global": ("P", "brown"),
    "dm_step_global": ("X", "darkorange"),
    "solar_bad": ("^", "orange"),
    "orbital_phase_bad": ("s", "blue"),
}

BAD_POINT_STYLE = ("x", "grey")
