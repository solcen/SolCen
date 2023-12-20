import csv
from glob import glob
import json
import os
import shutil
from classification import default_dump

ML_list = ['Knn', 'Decision_tree', 'Naive_bayes', 'Logistic_regression', 'Random_forest']
cent_list = ['cent_harm','cent_eigen','cent_close','cent_between','cent_degree']
FN = './res/FN/'
FP = './res/FP/'
TN = './res/TN/'
TP = './res/TP/'
test_index = {}
test_compare = {}
vul_names = []
# load result
for i in range(1,6):
    with open("contract_pred"+str(i)+".json","r") as f :
        tmp_dict=json.load(f)
        for k,v in tmp_dict.items():
            if k in test_index.keys():
                for k_cent,v_algm in v.items():
                    for k_algm,v_value in v_algm.items():
                        test_index[k][k_cent][k_algm] = test_index[k][k_cent][k_algm] or v_value
            else :
                test_index[k] = v
# load vul sol names
with open('testvul.json','r') as f:
        i = 0
        j = 0
        while True:
            text = f.readline()
            if text:
                text_line = json.loads(text)
                vul_names.append(text_line['file_id'].split('.sol')[0].rsplit('-',1)[0].split('_',1)[-1])
            else:
                break
res = {}

res["cent_harm"] = {}
res["cent_eigen"] = {}
res["cent_close"] = {}
res["cent_between"] = {}
res["cent_degree"] = {}
for ML_choice in ML_list :
    res["cent_harm"][ML_choice] = [0] * 4
    res["cent_eigen"][ML_choice] = [0] * 4
    res["cent_close"][ML_choice] = [0] * 4
    res["cent_between"][ML_choice] = [0] * 4
    res["cent_degree"][ML_choice] = [0] * 4
total_res = [0] * 4
flag = 0
for root, dirs, files in os.walk(TP, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
for root, dirs, files in os.walk(TN, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
for root, dirs, files in os.walk(FN, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
for root, dirs, files in os.walk(FP, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
#Calculation results of predict
for k,v in test_index.items():
    flag = 0
    contract_name = k.split('_',1)[-1]
    true_type = k.split('_',1)[0]
    for t,q in test_index[k].items() :
        if t not in cent_list :
            continue
        for ML_choice in ML_list :
            if (test_index[k][t][ML_choice] == 1) and (k in vul_names):
                res[t][ML_choice][0] += 1
                path_name = TP+t+'_'+ML_choice+'_'+k
                f = open(path_name,'w')
                f.write('')
                f.close()
            if (test_index[k][t][ML_choice] == 0) and (k not in vul_names):
                res[t][ML_choice][1] += 1
                path_name = TN+t+'_'+ML_choice+'_'+k
                f = open(path_name,'w')
                f.write('')
                f.close()
            if (test_index[k][t][ML_choice] == 1) and (k not in vul_names):
                res[t][ML_choice][2] += 1
                path_name = FP+t+'_'+ML_choice+'_'+k
                f = open(path_name,'w')
                f.write('')
                f.close()
            if (test_index[k][t][ML_choice] == 0) and (k in vul_names):
                res[t][ML_choice][3] += 1
                path_name = FN+t+'_'+ML_choice+'_'+k
                f = open(path_name,'w')
                f.write('')
                f.close()
with open("contract_pred_1.json","w") as f :
    json.dump(res,f,ensure_ascii=False, default=default_dump)
def modify(algorithm,TP,TN,FP,FN):
    Pre = TP/(TP + FP)
    Rec = TP/(TP + FN)
    Acc = (TP + TN)/(TP+ FP + TN + FN)
    F1 = 2*(Pre*Rec)/ (Pre+Rec)
    return [algorithm,0,F1,Pre,Rec,Acc]
with open("result_contract.csv",'w',newline='') as writer:
    csv_writer = csv.writer(writer)
    for k,v in res.items():
        csv_writer.writerow(str(k))
        csv_writer.writerow(['ML_Algorithm, Time, F1, Precision, Recall, Accuracy'])
        for k1,v1 in v.items():
            result = modify(k,v1[0],v1[1],v1[2],v1[3])
            csv_writer.writerow(result)
            

