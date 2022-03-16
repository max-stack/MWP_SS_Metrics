from difflib import SequenceMatcher
import sys
import json
sys.path.insert(0, '../mwp_solver')
from constants import TRAINSET_PATH, TESTSET_PATH, VALIDSET_PATH

class Similarity_Measure:
    def __init__(self):
        pass

    def get_nl(self, file):
        question_list = []
        with open(file) as json_file:
            data = json.load(json_file)

        for question in data:
            question_list.append(question["original_text"])
        
        return question_list
    
    def average_similarity(self):
        test_nl = self.get_nl(TESTSET_PATH)
        train_nl = self.get_nl(TRAINSET_PATH)

        added_similarities = 0
        for test_question in test_nl:
            max_ratio = 0
            for train_question in train_nl:
                max_ratio = max(max_ratio, SequenceMatcher(None, test_question, train_question).ratio())
            added_similarities += max_ratio
        
        result = {}

        result["similarity_score"] = added_similarities / len(test_nl)

        return result

# test = Similarity_Measure()
# print(test.average_similarity())