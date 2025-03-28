"""General definitions."""

# https://mne.tools/stable/documentation/glossary.html#term-data-channels
data_channel_types = [
    "mag",
    "grad",
    "eeg",
    "csd",
    "seeg",
    "ecog",
    "dbs",
    "hbo",
    "hbr",
    "fnirs_cw_amplitude",
    "fnirs_fd_ac_amplitude",
    "fnirs_fd_phase",
    "fnirs_od",
]

# https://mne.tools/stable/documentation/glossary.html#term-non-data-channels
non_data_channel_types = [
    "eog",
    "ecg",
    "resp",
    "emg",
    "ref_meg",
    "misc",
    "stim",
    "chpi",
    "exci",
    "ias",
    "syst",
    "bio",
    "temperature",
    "gsr",
    "gof",
    "dipole",
    "eyegaze",
    "pupil",
    "whitened",
]

microvolts = ("microvolt", "microvolts", "uV", "µV", "μV")
