from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class Channel_Delays:
    name: str
    channel_origin: int
    channel_delays: Dict[int, float]
    connections_info: Dict[Tuple[int, int], Dict[str, float]]

Run018 = Channel_Delays(
    name="Run018",
    channel_origin=8,
    channel_delays={
        0: 6.625,
        1: 7.671,
        2: 7.613,
        3: 7.46,
        4: 7.741,
        5: 7.275,
        6: 7.589,
        7: 7.421,
        8: 0.0,
        9: -2.584,
    },
    connections_info={
        (8, 4): {"delta_t": 7.741, "Amp[Cts]": 340, "degree": 1},
        (8, 7): {"delta_t": 7.421, "Amp[Cts]": 150, "degree": 1},
        (8, 9): {"delta_t": -2.584, "Amp[Cts]": 220, "degree": 1},
        (9, 6): {"delta_t": 10.173, "Amp[Cts]": 120, "degree": 2},
        (7, 1): {"delta_t": 0.25, "Amp[Cts]": 170, "degree": 2},
        (5, 4): {"delta_t": 0.466, "Amp[Cts]": 75, "degree": 2},
        (4, 3): {"delta_t": -0.281, "Amp[Cts]": 120, "degree": 2},  # corrigido
        (2, 1): {"delta_t": 0.058, "Amp[Cts]": 100, "degree": 3},
        (1, 0): {"delta_t": -1.046, "Amp[Cts]": 60, "degree": 3},
    }
)

Run013_J = Channel_Delays(
    name="Run013_J",
    channel_origin=16,
    channel_delays={
        14: -0.961,
        13: -0.843,
        23: -0.717,
        12: -0.687,
        10: -0.640,
        11: -0.567,
        8:  -0.439,
        9:  -0.541,
        18: -0.266,
        17: -0.233,
        19: -0.326,
        3:  -0.176,
        7:  -0.166,
        4:  -0.108,
        5:  -0.038,
        16: 0.0,
        0:  0.0,
        6:  0.032,
        2:  0.032,
        1:  0.086,
    },
    connections_info={
        (16, 8):  {"delta_t": -0.439, "Amp[Cts]": 150, "degree": 1},
        (16, 6):  {"delta_t": 0.032,  "Amp[Cts]": 600, "degree": 1},
        (16, 5):  {"delta_t": -0.038, "Amp[Cts]": 125, "degree": 1},

        (5, 7):   {"delta_t": -0.128, "Amp[Cts]": 60,  "degree": 2},
        (7, 17):  {"delta_t": -0.067, "Amp[Cts]": 60,  "degree": 3},

        (16, 18): {"delta_t": -0.266, "Amp[Cts]": 125, "degree": 1},
        (18, 19): {"delta_t": -0.060,  "Amp[Cts]": 40,  "degree": 2},
        (19, 9):  {"delta_t": -0.215, "Amp[Cts]": 250, "degree": 3},

        (18, 1):  {"delta_t": 0.352,  "Amp[Cts]": 200, "degree": 2},
        (1, 10):  {"delta_t": -0.726, "Amp[Cts]": 30,  "degree": 3},
        (1, 11):  {"delta_t": -0.653, "Amp[Cts]": 650, "degree": 3},
        (1, 0):   {"delta_t": -0.086, "Amp[Cts]": 60,  "degree": 3},
        (1, 2):   {"delta_t": -0.054, "Amp[Cts]": 125, "degree": 3},
        (2, 12):  {"delta_t": -0.719, "Amp[Cts]": 60,  "degree": 4},
        (12, 13): {"delta_t": -0.156, "Amp[Cts]": 10,  "degree": 5},
        (13, 14): {"delta_t": -0.118, "Amp[Cts]": 6,   "degree": 6},

        (2, 23):  {"delta_t": -0.749, "Amp[Cts]": 4,   "degree": 4},

        (1, 3):   {"delta_t": -0.262, "Amp[Cts]": 40,  "degree": 3},
        (3, 4):   {"delta_t": 0.068,  "Amp[Cts]": 43,  "degree": 4},

    }
)



DELAY_REGISTRY = {
    "Run018": Run018,
    "Run013_J": Run013_J,   # add future runs here
}