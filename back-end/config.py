from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """
    This class represents our project. It stores useful information about the structure, e.g. paths.
    """
    # PATHS
    base_dir: Path = Path(__file__).parents[0]

    checkpoint_dir = base_dir / 'checkpoint'
    DATASET_PATH = '../back-end/DataSet/face_age'

    # PARAMS IMG
    NUM_CLASSES = 24
    IMG_HEIGHT = 200
    IMG_WIDTH = 200
    CHANELS = 3

    # PARAMS NEURAL NETWORK
    BATCH_SIZE = 32
    SPLIT_PERCENTAGE = 0.8
    EPOCHS = 100
    LEARNING_RATE = 0.0001

    # (iniAge, endAge) , TAG
    RULES_BALANCER = [
        ((1, 2), 1),
        ((3, 5), 4),
        ((6, 8), 7),
        ((9, 11), 10),
        ((12, 14), 13),
        ((15, 17), 16),
        ((18, 20), 19),
        ((21, 23), 22),
        ((24, 26), 25),
        ((27, 29), 28),
        ((30, 32), 31),
        ((33, 35), 34),
        ((36, 38), 37),
        ((39, 41), 40),
        ((42, 44), 43),
        ((45, 47), 46),
        ((48, 50), 49),
        ((51, 53), 52),
        ((54, 56), 55),
        ((57, 59), 58),
        ((60, 64), 62),
        ((65, 70), 67),
        ((71, 76), 73),
        ((77, 80), 78),
        ((80, 90), 85)
    ]

    def __post_init__(self):
        # create the directories if they don't exist
        self.checkpoint_dir.mkdir(exist_ok=True)