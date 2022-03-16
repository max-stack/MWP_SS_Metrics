from constants import TRAINSET_PATH, TESTSET_PATH, VALIDSET_PATH
import json

class Create_Datasets:
    def __init__(self, test_data, valid_data=None, train_data=None):
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data

    def create_train_data(self):
        if(self.train_data != None):
            data = []
            id = 0
            for i in range(len(self.train_data[0])):
                question = {}
                question["id"] = id
                question["original_text"] = self.train_data[0][i]
                question["equation"] = self.train_data[1][i]
                question["ans"] = self.train_data[2][i]
                data.append(question)
                id += 1
            
            with open(TRAINSET_PATH, "w", encoding="utf-8") as data_file:
                json.dump(data, data_file, ensure_ascii=False, indent=4)
    
    def create_test_data(self):
        data = []
        id = 0
        for i in range(len(self.test_data[0])):
            question = {}
            question["id"] = id
            question["original_text"] = self.test_data[0][i]
            question["equation"] = self.test_data[1][i]
            question["ans"] = self.test_data[2][i]
            data.append(question)
            id += 1
        
        with open(TESTSET_PATH, "w", encoding="utf-8") as data_file:
            json.dump(data, data_file, ensure_ascii=False, indent=4)
        
    def create_valid_data(self):
        if(self.valid_data != None):
            data = []
            id = 0
            for i in range(len(self.valid_data[0])):
                question = {}
                question["id"] = id
                question["original_text"] = self.valid_data[0][i]
                question["equation"] = self.valid_data[1][i]
                question["ans"] = self.valid_data[2][i]
                data.append(question)
                id += 1
            
            with open(VALIDSET_PATH, "w", encoding="utf-8") as data_file:
                json.dump(data, data_file, ensure_ascii=False, indent=4)

