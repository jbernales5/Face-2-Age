from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    """
    This class represents our project. It stores useful information about the structure, e.g. paths.
    """
    # PATHS
    base_dir: Path = Path(__file__).parents[0]

    checkpoint_dir = base_dir / 'checkpoint'
    DATASET_PATH = '../DataSet/face_age'

    # PARAMS IMG
    NUM_CLASSES = 24
    IMG_HEIGHT = 200
    IMG_WIDTH = 200
    CHANELS = 3

    # PARAMS NEURAL NETWORK
    BATCH_SIZE = 1
    SPLIT_PERCENTAGE = 0.8
    EPOCHS = 100



    def __post_init__(self):
        # create the directories if they don't exist
        self.checkpoint_dir.mkdir(exist_ok=True)