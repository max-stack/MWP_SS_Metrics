DATASET = '../dataset/'

from quick_start import run_toolkit
from create_datasets import Create_Datasets

class MWP_Solver:
    def __init__(self, task_type="single_equation"):
        self.task_type = task_type


    def train_solver(self, train_data, test_data, valid_data):
        config_dict = {}

        dataset_creator = Create_Datasets(test_data, valid_data, train_data)
        dataset_creator.create_train_data()
        dataset_creator.create_test_data()
        dataset_creator.create_valid_data()

        results = {}
        
        results["Graph2Tree_Train"] = run_toolkit("Graph2Tree", self.task_type, config_dict)
        results["SAUSolver_Train"] = run_toolkit("SAUSolver", self.task_type, config_dict)

        return results

    def test_solver(self, test_data=None, set_created=False):
        config_dict = {}

        if(not set_created):
            dataset_creator = Create_Datasets(test_data)
            dataset_creator.create_test_data()

        results = {}
        
        results["Graph2Tree_Test"] = run_toolkit("Graph2Tree", self.task_type, config_dict, test_only=True)
        results["SAUSolver_Test"] = run_toolkit("SAUSolver", self.task_type, config_dict, test_only=True)

        return results



# if __name__ == '__main__':
#     config_dict = {}
#     run_toolkit("Graph2Tree", DATASET, "single_equation", config_dict)


# from constants import TRAINSET_PATH, TESTSET_PATH, VALIDSET_PATH
# import json

# train_data = []
# train_data.append([])
# train_data.append([])
# train_data.append([])
# test_data = []
# test_data.append([])
# test_data.append([])
# test_data.append([])
# valid_data = []
# valid_data.append([])
# valid_data.append([])
# valid_data.append([])

# with open(TRAINSET_PATH) as json_file:
#     data = json.load(json_file)

# for i in range(len(data)):
#     train_data[0].append(data[i]["original_text"])
#     train_data[1].append(data[i]["equation"])
#     train_data[2].append(data[i]["ans"])

# with open(TESTSET_PATH) as json_file:
#     data = json.load(json_file)

# for i in range(len(data)):
#     test_data[0].append(data[i]["original_text"])
#     test_data[1].append(data[i]["equation"])
#     test_data[2].append(data[i]["ans"])

# with open(VALIDSET_PATH) as json_file:
#     data = json.load(json_file)

# for i in range(len(data)):
#     valid_data[0].append(data[i]["original_text"])
#     valid_data[1].append(data[i]["equation"])
#     valid_data[2].append(data[i]["ans"])

# solver = MWP_Solver()
# print(solver.train_solver(train_data, test_data, valid_data))

