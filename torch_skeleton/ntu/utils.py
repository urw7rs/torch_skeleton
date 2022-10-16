import os.path as osp

_LABEL_MAP = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person’s stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person’s ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}

_TRAIN_SUBJECTS = [
    int(c)
    for c in (
        "1,2,4,5,8,9,"
        "13,14,15,16,17,18,19,"
        "25,27,28,"
        "31,34,35,38,"
        "45,46,47,49,"
        "50,52,53,54,55,56,57,58,59,"
        "70,74,78,"
        "80,81,82,83,84,85,86,89,"
        "91,92,93,94,95,97,98,"
        "100,103"
    ).split(",")
]

_TRAIN_CAMERAS = [2, 3]


def get_filename(file_path: str) -> str:
    filename: str = osp.basename(file_path).split(".")[0]
    return filename


def _field(filename: str, start: int, end: int):
    return int(filename[start:end])


def setup_from_name(filename: str) -> int:
    return _field(filename, 1, 4)


def camera_from_name(filename: str) -> int:
    return _field(filename, 5, 8)


def subject_from_name(filename: str) -> int:
    return _field(filename, 9, 12)


def label_from_name(filename: str) -> int:
    return _field(filename, 17, 20) - 1


def ntu_train_subjects():
    return _TRAIN_SUBJECTS


def ntu_train_cameras():
    return _TRAIN_CAMERAS


def class_to_label(idx):
    return _LABEL_MAP[idx]


_NTU_EDGE_INDEX = [
    [0, 1],
    [1, 20],
    [2, 20],
    [3, 2],
    [4, 20],
    [5, 4],
    [6, 5],
    [7, 6],
    [8, 20],
    [9, 8],
    [10, 9],
    [11, 10],
    [12, 0],
    [13, 12],
    [14, 13],
    [15, 14],
    [16, 0],
    [17, 16],
    [18, 17],
    [19, 18],
    [21, 22],
    [22, 7],
    [23, 24],
    [24, 11],
]


def edge_index():
    return _NTU_EDGE_INDEX


_MISSING_FILES = []


def ntu_missing_files():
    return _MISSING_FILES
