from difflib import SequenceMatcher
import json
import sys
sys.path.insert(0, '../mwp_solver')
from run_solver import MWP_Solver
from constants import TRAINSET_PATH, TESTSET_PATH, VALIDSET_PATH

class Test_Unsimilar:
    def __init__(self, similar_ratio=0.8):
        self.similar_ratio = similar_ratio
    
    def remove_similar(self):
        new_testset = []

        with open(TESTSET_PATH) as test_file:
            test_data = json.load(test_file)
        
        with open(TRAINSET_PATH) as train_file:
            train_data = json.load(train_file)
        
        train_questions = [x["original_text"] for x in train_data]

        i=0
        for test_question in test_data:
            max_ratio = 0
            test_nl = test_question["original_text"]
            for train_nl in train_questions:
                max_ratio = max(max_ratio, SequenceMatcher(None, test_nl, train_nl).ratio())
            if max_ratio < self.similar_ratio:
                new_testset.append(test_question)
            i += 1
            print(i)

        with open(TESTSET_PATH, "w", encoding="utf-8") as testset_file:
            json.dump(new_testset, testset_file, ensure_ascii=False, indent=4)
    
    def test_unsimilar(self):
        self.remove_similar()
        solver = MWP_Solver()
        test_results = solver.test_solver(set_created=True)

        results = {}
        
        results["Graph2Tree_Test_Unsimilar"] = test_results["Graph2Tree_Test"]
        results["SAUSolver_Test_Unsimilar"] = test_results["SAUSolver_Test"]

        return results

# test1 = MWP_Solver()
# print(test1.test_solver(set_created=True))
# test = Test_Unsimilar()
# print(test.test_unsimilar())