from pathlib import Path

DATASET_PATH = str(Path(__file__).parent / "../dataset")
TRAINSET_PATH = DATASET_PATH + "/trainset.json"
TESTSET_PATH = DATASET_PATH + "/testset.json"
VALIDSET_PATH = DATASET_PATH + "/validset.json"