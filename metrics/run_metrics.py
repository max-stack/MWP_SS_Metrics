import sys
sys.path.insert(0, '../mwp_solver')
from run_solver import MWP_Solver
sys.path.insert(0, '../similarity_test')
from similarity_measure import Similarity_Measure
from test_unsimilar import Test_Unsimilar

class Run_Metrics:
    def __init__(self, task_type="single_equation"):
        self.task_type = task_type

    def init_model(self, train_data, test_data, valid_data):
        solver = MWP_Solver(task_type=self.task_type)
        self.train_metric = solver.train_solver(train_data, test_data, valid_data)
    
    def get_metrics(self, test_data, similar_ratio=0.8):
        test_solver = MWP_Solver(task_type=self.task_type)
        test_metric = test_solver.test_solver(test_data)

        similarity = Similarity_Measure()
        similarity_metric = similarity.average_similarity()
        
        test_unsimilar = Test_Unsimilar()
        test_unsimilar_metric = test_unsimilar.test_unsimilar(similarity_ratio)
        
        final_metric = {}
        final_metric.uodate(self.train_metric)
        final_metric.update(test_metric)
        final_metric.update(similarity_metric)
        final_metric.update(test_unsimilar_metric)
        
        return result

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

# with open(TRAINSET_PATH, "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)

# for i in range(len(data)):
#     train_data[0].append(data[i]["original_text"])
#     train_data[1].append(data[i]["equation"])
#     train_data[2].append(data[i]["ans"])

# with open(TESTSET_PATH, "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)

# for i in range(len(data)):
#     test_data[0].append(data[i]["original_text"])
#     test_data[1].append(data[i]["equation"])
#     test_data[2].append(data[i]["ans"])

# with open(VALIDSET_PATH, "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)

# for i in range(len(data)):
#     valid_data[0].append(data[i]["original_text"])
#     valid_data[1].append(data[i]["equation"])
#     valid_data[2].append(data[i]["ans"])

# test = Run_Metrics()
# test.init_model(train_data, test_data, valid_data)
# import os
# os.remove("../dataset/deprel_tree_info.json")

# test_data = [
#     [
#     "there are 3.0 crayons in 4.0 boxes . how many crayons are there in all ?",
#     "at the town carnival billy rode the ferris wheel 3.0 times and the bumper cars 7.0 times . if each ride cost 3.0 tickets , how many tickets did he use ?",
#     "jason went to the mall to buy clothes . she bought a football for $ 9.31 , a strategy game for $ 11.08 , and a song book for $ 14.33 . how much did jason spend on video games ?",
#     "there are 12 crayons in the drawer . mary placed 41 crayons out of the drawer . how many crayons are now there in total ?",
#     "lemon heads come in packages of 256.0 . louis ate 32.0 lemon heads . how many cases of 256.0 boxes , plus extra boxes does jenny need to deliver ?",
#     "ned was trying to expand his game collection . he bought 23.0 games from a friend and bought 21.0 more at a garage sale . if 8.0 of the games did n't work , how many good games did he end up with ?",
#     "lewis saved checking on the grapevines for his collection . he needs 12 ounces of 12.50 hours . how much money will he make ?",
#     "melanie had pennies and 2 pennies in her bank . her dad gave her 7 pennies and her mother gave her 9 pennies . how many pennies does melanie have now ?",
#     "in mr. olsen 's mathematics complex , 0.25 the garments are bikinis and 0.375 are trunks . what fraction of the garments are either bikinis or trunks ?",
#     "freeport mcmoran projects that in the library . she has 7 bracelets with 140 cents per hour . how many gallons of gas did she need to buy the notebook ?",
#     "mary has 25 cupcakes . her friend has 54 . how many more cupcakes do you have ?",
#     "while playing a trivia game , adam answered 8.0 questions correct in the first half and 4.0 questions correct in the second half . if each question was worth 2.0 points , what was his total ?"
#     ],
#     [
#     "X=(4.0*3.0)",
#     "X=(3.0*(7.0+3.0))",
#     "X=11.08+14.33+9.31",
#     "X=41+12",
#     "X=(256.0/32.0)",
#     "X=((21.0+8.0)-23.0)",
#     "x=12.50*12",
#     "X=7+9+2",
#     "x=0.375+0.25",
#     "x=140/7",
#     "x=54-(15+25)",
#     "X=(8.0*(4.0+2.0))"
#     ],
#     [
#     12.0,
#     30.0,
#     "34.72",
#     "53",
#     8.0,
#     6.0,
#     150.0,
#     "18",
#     0.625,
#     20.0,
#     14.0,
#     48.0
#     ]
# ]
# print(test.get_metrics(test_data))