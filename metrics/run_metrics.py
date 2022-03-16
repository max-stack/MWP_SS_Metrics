import sys
sys.path.insert(0, '../mwp_solver')
from run_solver import MWP_Solver
sys.path.insert(0, '../similarity_test')
from similarity_measure import Similarity_Measure
from test_unsimilar import Test_Unsimilar

class Run_Metrics:
    def __init__(self, task_type="single_equation"):
        self.task_type = task_type
        self.train_metric = {}

    def init_model(self, train_data, test_data, valid_data):
        solver = MWP_Solver(task_type=self.task_type)
        self.train_metric = solver.train_solver(train_data, test_data, valid_data)
    
    def get_metrics(self, test_data, similar_ratio=0.8):
        test_solver = MWP_Solver(task_type=self.task_type)
        test_metric = test_solver.test_solver(test_data)

        similarity = Similarity_Measure()
        similarity_metric = similarity.average_similarity()
        
        test_unsimilar = Test_Unsimilar(similar_ratio)
        test_unsimilar_metric = test_unsimilar.test_unsimilar()
        
        final_metric = {}
        final_metric.update(self.train_metric)
        final_metric.update(test_metric)
        final_metric.update(similarity_metric)
        final_metric.update(test_unsimilar_metric)
        
        return final_metric

from constants import TRAINSET_PATH, TESTSET_PATH, VALIDSET_PATH
import json

train_data = []
train_data.append([])
train_data.append([])
train_data.append([])
test_data = []
test_data.append([])
test_data.append([])
test_data.append([])
valid_data = []
valid_data.append([])
valid_data.append([])
valid_data.append([])

with open(TRAINSET_PATH, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

for i in range(len(data)):
    train_data[0].append(data[i]["original_text"])
    train_data[1].append(data[i]["equation"])
    train_data[2].append(data[i]["ans"])

with open(TESTSET_PATH, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

for i in range(len(data)):
    test_data[0].append(data[i]["original_text"])
    test_data[1].append(data[i]["equation"])
    test_data[2].append(data[i]["ans"])

with open(VALIDSET_PATH, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

for i in range(len(data)):
    valid_data[0].append(data[i]["original_text"])
    valid_data[1].append(data[i]["equation"])
    valid_data[2].append(data[i]["ans"])

test = Run_Metrics()
test.init_model(train_data, test_data, valid_data)
import os
os.remove("../dataset/deprel_tree_info.json")

test_data = [
    [
    "there are 3.0 crayons in 4.0 boxes . how many crayons are there in all ?",
    "at the town carnival billy rode the ferris wheel 3.0 times and the bumper cars 7.0 times . if each ride cost 3.0 tickets , how many tickets did he use ?",
    "jason went to the mall to buy clothes . she bought a football for $ 9.31 , a strategy game for $ 11.08 , and a song book for $ 14.33 . how much did jason spend on video games ?",
    "there are 12 crayons in the drawer . mary placed 41 crayons out of the drawer . how many crayons are now there in total ?",
    "lemon heads come in packages of 256.0 . louis ate 32.0 lemon heads . how many cases of 256.0 boxes , plus extra boxes does jenny need to deliver ?",
    "ned was trying to expand his game collection . he bought 23.0 games from a friend and bought 21.0 more at a garage sale . if 8.0 of the games did n't work , how many good games did he end up with ?",
    "lewis saved checking on the grapevines for his collection . he needs 12 ounces of 12.50 hours . how much money will he make ?",
    "melanie had pennies and 2 pennies in her bank . her dad gave her 7 pennies and her mother gave her 9 pennies . how many pennies does melanie have now ?",
    "in mr. olsen 's mathematics complex , 0.25 the garments are bikinis and 0.375 are trunks . what fraction of the garments are either bikinis or trunks ?",
    "freeport mcmoran projects that in the library . she has 7 bracelets with 140 cents per hour . how many gallons of gas did she need to buy the notebook ?",
    "mary has 25 cupcakes . her friend has 54 . how many more cupcakes do you have ?",
    "while playing a trivia game , adam answered 8.0 questions correct in the first half and 4.0 questions correct in the second half . if each question was worth 2.0 points , what was his total ?"
    "my car gets 5.0 miles per gallon of gas . how many bags of she did she pick ?",
    "annie has blocks . ann has with 29.0 blocks . ann finds another 9.0 . how many blocks does ann end with ?",
    "a pet store had 9.0 puppies . in one day they sold 102.0 of them and put the rest into cages with 21.0 in each cage . how many cages did they use ?",
    "mrs. hilt saw 71 bees in the hive . the next day she saw 6 times that many . how many bees did she begin with ?",
    "there are 8 pencils in the classroom , each student will have 3 pencils . how many pencils are there in all ?",
    "4.0 ducks are swimming in a lake . 2.0 more ducks come to join them . how many ducks are swimming in the lake ?",
    "a tailor cut 0.5 of an inch off a skirt and 0.75 of an inch off a pair of pants . how much more did the tailor cut off the skirt than the pants ?",
    "a car rents for 1500.0 dollars per day plus 1250.0 cents per mile . you are on a daily budget of 0.01 dollars and 1250.0 cents per mile . how many hours will she make ?",
    "sunshine car rentals rents a basic car at a daily rate of 79.0 dollars plus 15.0 cents per mile . city rentals rents a basic car at 1.0 dollars plus 15.0 cents per mile . for how many cents will he make ?",
    "frank and her friends were recycling paper for their class . for every 24.0 pounds they recycled they earned one point . if haley recycled 3.0 pounds and her friends recycled 3.0 pounds , how many points did they earn ?",
    "fred grew 816.0 pennies in her bank . her sister borrowed 1339.0 of a gallon of sugar and now has 816.0 off . how many pounds of fruit did janet 's sandcastle than her sister ?",
    "sara picked 11 pears and sally picked 45 pears from the pear tree . how many pears were picked in total ?",
    "on friday , the students received 2 feet of 12 students went on the zoo . what will the total number will be in all ?",
    "zoe had 11.0 songs on her mp3 player . if she deleted 42.0 old songs from it and then added 27.0 new songs , how many songs does she have on her mp3 player ?",
    "blake filled a bucket with 35 of a gallon of gas . she already put 83 cups of flour on each package . how many cups of flour does she have now ?",
    "the difference of 4.0 times a number and 8.0 is 17.0 . find the number .",
    "joan had 86 peaches at her roadside fruit dish . he went to the orchard and picked peaches to stock up . there are now 34 peaches . how many did he pick ?",
    "90.0 painter can paint a house in 810.0 hours , how many hours will it take for the job in still air ?"
    ],
    [
    "X=(4.0*3.0)",
    "X=(3.0*(7.0+3.0))",
    "X=11.08+14.33+9.31",
    "X=41+12",
    "X=(256.0/32.0)",
    "X=((21.0+8.0)-23.0)",
    "x=12.50*12",
    "X=7+9+2",
    "x=0.375+0.25",
    "x=140/7",
    "x=54-(15+25)",
    "X=(8.0*(4.0+2.0))",
    "X=(60.0*5.0)",
    "X=(9.0+29.0)",
    "X=((102.0-21.0)/9.0)",
    "x=6*71",
    "x=3*8",
    "X=(2.0+4.0)",
    "X=0.75-0.5",
    "x=(1500.0-1250.0)/(5*0.01)",
    "x=(79.0-15.0)/(4.0-1.0)",
    "X=((3.0+24.0)/3.0)",
    "X=1339.0-816.0",
    "X=45+11",
    "x=(12-8)*2",
    "X=((27.0+42.0)-11.0)",
    "x=83-35",
    "x=(17.0+4.0)/8.0",
    "x=86-34",
    "x=810.0/((810.0/135.0)+(810.0/90.0))"
    ],
    [
    12.0,
    30.0,
    "34.72",
    "53",
    8.0,
    6.0,
    150.0,
    "18",
    0.625,
    20.0,
    14.0,
    48.0,
    300.0,
    38.0,
    9.0,
    426.0,
    24.0,
    6.0,
    "0.25",
    5000.0,
    21.3333,
    9.0,
    "523.0",
    "56",
    8.0,
    58.0,
    48.0,
    2.625,
    52.0,
    54.0
    ]
]
print(test.get_metrics(test_data))