import os
import re
import numpy as np
import json
import pandas as pd



extract_information_2 = ["MRR", "MR", "HITS@1", "HITS@3", "HITS@10", "fmeasure", "accuracy","Namespace(K=1,"]
information_3 = ['lr', 'gamma', 'hiddim', 'randvar', 'batch_size']
data_dir = "models/SDE"

test_file_names = ['Test_orig', 'Test_fmeasure', 'Test_1_tail', 'Test_hit_1', 'Test_hit_ten', 'Test_set_four', 'Test_set_five', 'Test_set_current_metrics', 'Test_set_to_explore']

all_files = np.zeros((1,15))

all_hyperparams = np.zeros((1,12))

#/home/turzo/PycharmProjects/RotatE_Updated/models/TransE
for test_file_name in test_file_names:
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            #file_entries = np.empty([1, 20])
            if file.endswith("*.text"):
                #print(os.path.join(root, file))
                #print(file)
                if (open(os.path.join(root, file), 'r').read().find('Hit@10')) != -1:
                    file = open(os.path.join(root, file), 'r')

                    Lines = file.readlines()

                    #entries_for_file = np.zeros((1,3))
                    entries_for_file = np.zeros((1,3))
                    #other_infos = np.zeros((1,5))
                    # fmeasure, accuracy, avg, left, right = 0, 0, 0, 0, 0
                    # model = ''
                    # hyps = ''
                    # hyperparams = []
                    line_entries = np.empty((0,1))
                    for line in Lines:
                        #print(line)
                        for var in extract_information_2:
                            #param = test_file_name + ' Namespace'
                            #print(line.split())
                            if (var in line.split()) and (test_file_name in line):
                                elements = line.split()
                                #print(elements)
                                #print(elements[8])
                                #fmeasure = elements[7]

                                if test_file_name == 'Test_fmeasure' or test_file_name == 'Test_1_tail':

                                    if var == 'fmeasure':
                                        fmeasure = elements[8]
                                        line_entries = np.hstack((line_entries, fmeasure))
                                        print(line_entries)

                                    elif var == 'accuracy':
                                        accuracy = elements[8]
                                        line_entries = np.hstack((line_entries, accuracy))
                                    elif var in ["MRR", "MR", "HITS@1", "HITS@3", "HITS@10"]:
                                        avg, left, right = elements[8], elements[9], elements[10]
                                        line_entries = np.hstack((line_entries, avg))
                                        line_entries = np.hstack((line_entries, left))
                                        line_entries = np.hstack((line_entries, right))
                                    elif var == 'Namespace(K=1,':
                                        #hyp, gamma ,dimension, hiddim, hidrank, lr  = elements[6] , elements[17], elements[20], elements[21], elements[22], elements[24],
                                        #bs= re.findall(r'\b\d+\b', hyp)
                                        model = elements[28]
                                        line_entries = np.hstack((line_entries, model))
                                        #print(model)
                                        hyps = elements[4:]
                                        #print(hyps)
                                        #print(avg, left, right)
                                else:
                                    continue

                else:
                    continue
                print(line_entries)
            else:
                continue
                                # avg, left, right = elements[8], elements[9], elements[10]
                                # entry = np.array([avg, left, right]).reshape((1,3))
                                # #print(entry)
                                # #print(entry.shape)
                                # #print(entries_for_file.shape)
                                # #exit()
                                # #print(entries_for_file.shape)
                                # #print(entry.shape)
                                # #exit()
                                # entries_for_file = np.hstack((entries_for_file, entry))
                else:
                    continue
                entries_for_file = entries_for_file[:,3:]
                print(all_files.shape)
                print(entries_for_file.shape)
                #exit()
                all_files = np.vstack((all_files, entries_for_file))
                #print(entries_for_file)
                #exit()
            else:
                continue

        for file in files:
            #if (open(os.path.join(root, file), 'r').read().find('Test_orig')) != -1:
            #file_entries = np.empty([1, 20])
            if file.endswith(".json"):

                f = open(os.path.join(root, file), )

                json_logs = json.load(f)
                n_neg = json_logs['negative_sample_size']
                gamma = json_logs['gamma']
                adv_temp = json_logs['adversarial_temperature']
                model_name = json_logs['model']
                learning_rate = json_logs['learning_rate']
                regularization = json_logs['regularization']
                step_size = json_logs['max_steps']
                hidden_dim = json_logs['hidden_dim']
                batch_size = json_logs['batch_size']
                hiddim = json_logs['hiddim']
                randvar = json_logs['randvar']
                multdif = json_logs['multdif']

                hyperparams = np.array([n_neg , gamma , adv_temp, model_name, learning_rate, regularization, step_size, hidden_dim, batch_size, hiddim, randvar, multdif ]).reshape((1,12))
                all_hyperparams = np.vstack((all_hyperparams, hyperparams))
            else:
                continue

    all_files = np.delete(all_files, (0), axis=0)
    all_hyperparams = np.delete(all_hyperparams, (0), axis=0)
    to_save_in_excel = np.c_[all_files, all_hyperparams]
    to_save_in_excel_df = pd.DataFrame(to_save_in_excel)
    to_save_in_excel_df.columns = ['MRR_avg', 'MRR_left', 'MRR_right', 'MR_avg', 'MR_left', 'MR_right', 'Hit@1_avg', 'Hit@1_left', 'Hit@1_right'
                                   , 'Hit@3_avg', 'Hit@3_left', 'Hit@3_right', 'Hit@10_avg', 'Hit@10_left', 'Hit@10_right',
                                   'number of neg samples', 'gamma', 'adv_temp', 'model_name', 'learning_rate', 'regularization', 'step_size', 'hidden_dim',
                                   'batch_size', 'hiddim', 'randvar', 'multdif'
                                   ]

    save_data_dir = data_dir
    save_file_name = test_file_name + '_results_with_hyper_params_SDE.xlsx'
    write_path = os.path.join(data_dir, save_file_name)
    to_save_in_excel_df.to_excel(write_path, index = False, header=True)