from pathlib import Path

MODELS = ["Graph2Tree", "SAUSolver"]

DATASET_PATH = str(Path(__file__).parent / "../dataset")
TRAINSET_PATH = DATASET_PATH + "/trainset.json"
TESTSET_PATH = DATASET_PATH + "/testset.json"
VALIDSET_PATH = DATASET_PATH + "/validset.json"

LOG_PATH = str(Path(__file__).parent / "logs")
CHECKPOINT_PATH = str(Path(__file__).parent / "checkpoint")
TRAINED_MODEL_PATH = str(Path(__file__).parent / "trained_model")