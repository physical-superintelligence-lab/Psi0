import numpy as np

# fmt: off
# Hand skeletons
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9,10), (10,11), (11,12),     # Middle
    (0,13), (13,14), (14,15), (15,16),    # Ring
    (0,17), (17,18), (18,19), (19,20)     # Pinky
]

# Define colors for each finger
FINGER_COLORS = {
    "wrist": [1.0, 1.0, 1.0],  # white
    "thumb": (0.969, 0.882, 0.369),  # [1.0, 1.0, 0.0],    # yellow
    "index": (0.690, 0.227, 0.557),  # [0.5, 0.0, 0.5],    # purple
    "middle": (0.188, 0.259, 0.522),  # [0.0, 0.0, 1.0],    # blue
    "ring": (0.831, 0.310, 0.161),  # [1.0, 0.5, 0.0],    # dark orange
    "pinky": (0.549, 0.659, 0.294),  # [0.0, 1.0, 1.8]     # cyan
}

FINGER_JOINT_INDICES = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

PERSON_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 128, 128),
]
# fmt: on

def get_joint_colors():
    # Assign a color per joint
    joint_colors = np.zeros((21, 3), dtype=np.float32)
    for fname, ids in FINGER_JOINT_INDICES.items():
        for i in ids:
            joint_colors[i] = FINGER_COLORS[fname]
    return joint_colors
