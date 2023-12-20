import json
import os
import random
from glob import glob
import shutil
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
from time import time
import csv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")
trainnon_file = 'trainnon.json' #contracts with no-vul-label for train
trainvul_file = 'trainvul.json' #contracts with vul-label for train
testnon_file = 'testnon.json' #contracts with no-vul-label for test
testvul_file = 'testvul.json' #contracts with vul-label for test
result_path = './result4-fewertrainnon1-2-2.csv'  # temp savepath  # result0~4
# contract pred result is in ./contract_pred.json
cent_type_list = ['cent_eigen','cent_harm','cent_close', 'cent_between', 'cent_degree']
ML_list = ['Knn', 'Decision_tree', 'Naive_bayes', 'Logistic_regression', 'Random_forest']  # 机器学习模型列表
train_non_num = 2989 * 2 # double num of test-vul-func

class ml:
    def machine_learning(self,train_X, train_Y, test_X, test_Y, choice,pre_pred):
        
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)


        start_time = time()
        model = ML_list[choice] # choice machine learning
        print('model: ', model)
        if choice == 0:
            clf = KNeighborsClassifier()
        elif choice == 1:
            clf = DecisionTreeClassifier()
        elif choice == 2:
            clf = GaussianNB()
        elif choice == 3:
            clf = LogisticRegression()
        elif choice == 5:
            clf = SVC()
        elif choice == 4:
            clf = RandomForestClassifier()
        else:
            print("Choice Error!")
            return
        
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
        end_time = time()

        for i in range(len(y_pred)):
            if pre_pred[i] == 0:
                y_pred[i] = 0
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)
        del clf
        return end_time - start_time, f1, precision, recall, accuracy,y_pred


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

final_test_vul = []
final_test_non = []
list_pred = {}

max_dim = {}
final_train_non = []
final_train_vul = []

def main():
    with open(trainnon_file, 'r') as f:
        j = 0
        while True:
            text = f.readline()
            if text:
                text_line = json.loads(text)
                tmp = {}
                tmp['file_id'] = text_line['file_id']
                tmp['isgrey'] = text_line['isgrey']
                for cent_type in cent_type_list:
                    tmp[cent_type] = text_line[cent_type]
                final_train_non.append(tmp)
                del text_line
                del tmp
                j +=1
                print(j)
            else:
                break
            if j > train_non_num:  # train dataset: non func : vul func = 1:2
                break
    with open(trainvul_file,'r') as f:
        j = 0
        while True:
            text = f.readline()
            if text:
                text_line = json.loads(text)
                final_train_vul.append(text_line)
                j +=1
                
            else:
                break
    print("train file read")
    print()

    with open(testnon_file, 'r') as f:
        i = 0
        while True:
            text = f.readline()
            if text:
                text_line = json.loads(text)
                tmp = {}
                tmp['file_id'] = text_line['file_id']
                tmp['isgrey'] = text_line['isgrey']
                for cent_type in cent_type_list:
                    tmp[cent_type] = text_line[cent_type]
                final_test_non.append(tmp)
                for cent_type in cent_type_list:
                    if max_dim.__contains__(cent_type):
                        max_dim[cent_type] = max(max_dim[cent_type],len(text_line[cent_type]))
                    else :
                        max_dim[cent_type] = len(text_line[cent_type])
                del text_line
                del tmp
                i += 1
                
            else:
                break
    with open(testvul_file,'r') as f:
        i = 0
        j = 0
        while True:
            text = f.readline()
            if text:
                text_line = json.loads(text)
                final_test_vul.append(text_line)
                for cent_type in cent_type_list:
                    if max_dim.__contains__(cent_type):
                        max_dim[cent_type] = max(max_dim[cent_type],len(text_line[cent_type]))
                    else :
                        max_dim[cent_type] = len(text_line[cent_type])
                j +=1
                i +=1
            else:
                break


    print("test file read")

    
    # classification witth different centrality vectors and different MLs
    if os.path.exists(result_path):
        os.remove(result_path)
    f = open(result_path,'a')
    csv_writer = csv.writer(f)
    for cent_type in cent_type_list:
        print(cent_type)
        
        csv_writer.writerow([cent_type])
        csv_writer.writerow(['ML_Algorithm, Time, F1, Precision, Recall, Accuracy'])

        # extract vector from the centrality vector matrix and normalized to the same dimension
        vectors = []
        labels = []
        max_dimen = 0
        for cent_matrix in final_train_non:
            vector = cent_matrix[cent_type]
            vectors.append(vector)
            labels.append(0)
            if max_dimen < len(vector):
                max_dimen = len(vector)
        for cent_matrix in final_train_vul:
            vector = cent_matrix[cent_type]
            vectors.append(vector)
            labels.append(1)
            if max_dimen < len(vector):
                max_dimen = len(vector)
        max_dimen = max(max_dimen,max_dim[cent_type])
        for vector in vectors:
            for j in range(len(vector), max_dimen):
                vector.append(0)
        train_vectors = []
        train_labels = []
        test_vectors = []
        test_labels = []
        test_pre_pred = []
        for index in range(len(vectors)):
            train_vectors.append(vectors[index])
            train_labels.append(labels[index])
        del vectors
        del labels
        
        for index in range(len(final_test_vul)):
            for j in range(len(final_test_vul[index].get(cent_type)), max_dimen):
                final_test_vul[index][cent_type].append(0)
            test_vectors.append(final_test_vul[index].get(cent_type))
            test_labels.append(1)
            test_pre_pred.append(-1)
        for index in range(len(final_test_non)):
            for j in range(len(final_test_non[index].get(cent_type)), max_dimen):
                final_test_non[index][cent_type].append(0)
            test_vectors.append(final_test_non[index].get(cent_type))
            test_labels.append(0)
            if final_test_non[index]['isgrey'] == 0:
                test_pre_pred.append(0)
            else :
                test_pre_pred.append(-1)
        # classification using MLs
        all_result = []
        for ML_choice in range(len(ML_list)):
            new_ml = ml()
            time_, f1, precision, recall, accuracy,labels_pred= new_ml.machine_learning(train_vectors, train_labels, test_vectors, test_labels,
                                                        ML_choice,test_pre_pred)
            del new_ml
            print("learn success")
            if cent_type not in list_pred:
                list_pred[cent_type] = {}
            list_pred[cent_type][ML_list[ML_choice]] = labels_pred
            del labels_pred
            result = [ML_list[ML_choice], time_, f1, precision, recall, accuracy]
            # save result
            print('ML_Algorithm, Time, F1, Precision, Recall, Accuracy')
            print(result)
            csv_writer.writerow(result)
        # release memory
        del train_vectors
        del train_labels
        del test_vectors
        del test_labels
        del test_pre_pred
    f.close()
    total = {}
    j = -1
    #Summarize the results and store them
    for index in range(len(final_test_vul)):
        j+=1
        contract_name = final_test_vul[index]["file_id"].split('.sol')[0].rsplit('-',1)[0].split('_',1)[-1]
        if total.__contains__(contract_name) :
            for cent_type in cent_type_list :
                for ML_choice in ML_list:
                        pred_array = list_pred[cent_type][ML_choice]
                        if cent_type not in total[contract_name]:
                            total[contract_name][cent_type] = {}
                        if total[contract_name][cent_type].__contains__(ML_choice):
                            total[contract_name][cent_type][ML_choice] = total[contract_name][cent_type][ML_choice] or pred_array[j]
                        else :
                            total[contract_name][cent_type][ML_choice] = pred_array[j]
        else :
            total[contract_name] = {}
            for cent_type in cent_type_list :
                t = 0
                for ML_choice in ML_list:
                        pred_array = list_pred[cent_type][ML_choice]
                        if cent_type not in total[contract_name]:
                            total[contract_name][cent_type] = {}
                        if total[contract_name][cent_type].__contains__(ML_choice):
                            total[contract_name][cent_type][ML_choice] = total[contract_name][cent_type][ML_choice] or pred_array[j]
                        else :
                            total[contract_name][cent_type][ML_choice] = pred_array[j]
    for index in range(len(final_test_non)):
        j+=1
        contract_name = final_test_non[index]["file_id"].split('.sol')[0].rsplit('-',1)[0]
        if total.__contains__(contract_name) :
            for cent_type in cent_type_list :
                for ML_choice in ML_list:
                    if total[contract_name][cent_type][ML_choice] == 1 :
                        continue
                    else :
                        t = 0
                        pred_array = list_pred[cent_type][ML_choice]
                        if cent_type not in total[contract_name]:
                            total[contract_name][cent_type] = {}
                        if total[contract_name][cent_type].__contains__(ML_choice):
                            total[contract_name][cent_type][ML_choice] = total[contract_name][cent_type][ML_choice] or pred_array[j]
                        else :
                            total[contract_name][cent_type][ML_choice] = pred_array[j]
        else :
            total[contract_name] = {}
            for cent_type in cent_type_list :
                t = 0
                for ML_choice in ML_list:
                        pred_array = list_pred[cent_type][ML_choice]
                        if cent_type not in total[contract_name]:
                            total[contract_name][cent_type] = {}
                        if total[contract_name][cent_type].__contains__(ML_choice):
                            total[contract_name][cent_type][ML_choice] = total[contract_name][cent_type][ML_choice] or pred_array[j]
                        else :
                            total[contract_name][cent_type][ML_choice] = pred_array[j]
    with open("contract_pred.json","w") as wf :
        json.dump(total,wf,ensure_ascii=False, default=default_dump)

if __name__ == '__main__':
    main()
