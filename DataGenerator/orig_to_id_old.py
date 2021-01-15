# # pos_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/_train.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # test_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/_test.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # validation_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/_test.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # #pos_triples = pos_triples[:,:-1]
# # data_train = pd.DataFrame(pos_triples, columns=["subject", "predicate", "object"])
# # data_test = pd.DataFrame(test_triples, columns=["subject", "predicate", "object"])
# # data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "object"])
# #
# # frames = [data_train, data_test, data_valid]
# # result = pd.concat(frames)
# # result_in_array = np.array(result)
# #
# # #train_pos, test_pos = train_test_split(pos_triples, test_size=0.2)
# # data = pd.DataFrame(pos_triples, columns=["subject", "predicate", "object"])
# #
# # entity_to_id, rel_to_id = create_mappings(triples=result_in_array)
# # mapped_pos_all_tripels, _, _ = create_mapped_triples(triples=pos_triples, entity_to_id=entity_to_id,
# #                                                                rel_to_id=rel_to_id)
# #
# # ent2id = dict((v,k) for k,v in entity_to_id.items())
# # rel2id = dict((v,k) for k,v in rel_to_id.items())
# # write_dic('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/entities.dict',ent2id)
# # write_dic('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/relations.dict',rel2id)
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/all_triples.txt',mapped_pos_all_tripels)
# #
# # mapped_pos_train_tripels, _, _ = create_mapped_triples(triples=np.array(data_train), entity_to_id=entity_to_id,
# #                                                                rel_to_id=rel_to_id)
# #
# # mapped_pos_test_tripels, _, _ = create_mapped_triples(triples=np.array(data_test), entity_to_id=entity_to_id,
# #                                                                rel_to_id=rel_to_id)
# #
# # mapped_pos_validation_tripels, _, _ = create_mapped_triples(triples=np.array(data_valid), entity_to_id=entity_to_id,
# #                                                                rel_to_id=rel_to_id)
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/original_quadropoles.txt',mapped_pos_train_tripels)
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/test.txt',mapped_pos_test_tripels)
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/validation.txt',mapped_pos_validation_tripels)
#
#
# #For wordnet RR, Assuming i already have the entiy to id file
# #############################################################
# # train_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/original_quadropoles.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # test_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/test.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # validation_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/valid.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# #
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/entities.dict', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/relations.dict', header=None)
# #
# # train = original_id_to_triples(train_triples, entity_to_id, rel_to_id)
# # test = original_id_to_triples(test_triples, entity_to_id, rel_to_id)
# # valid = original_id_to_triples(validation_triples, entity_to_id, rel_to_id)
# #
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/mapped/original_quadropoles.txt',np.array(train))
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/mapped/test.txt',np.array(test))
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/WN18rr/mapped/valid.txt',np.array(valid))
# #
# # #For DB100k. Here we do not have the mapping
# # pos_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/_train.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # test_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/_test.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # validation_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/_test.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# # #pos_triples = pos_triples[:,:-1]
# # data_train = pd.DataFrame(pos_triples, columns=["subject", "predicate", "object"])
# # data_test = pd.DataFrame(test_triples, columns=["subject", "predicate", "object"])
# # data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "object"])
#
#
# # frames = [data_train, data_test, data_valid]
# # result = pd.concat(frames)
# # result_in_array = np.array(result)
# #
# # #train_pos, test_pos = train_test_split(pos_triples, test_size=0.2)
# # data = pd.DataFrame(pos_triples, columns=["subject", "predicate", "object"])
# #
# # entity_to_id, rel_to_id = create_mappings(triples=result_in_array)
# # ent2id = dict((v,k) for k,v in entity_to_id.items())
# # rel2id = dict((v,k) for k,v in rel_to_id.items())
# #
# # write_dic('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/mapped/entities.dict',ent2id)
# # write_dic('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/mapped/relations.dict',rel2id)
# #
# #
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/mapped/entities.dict', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/mapped/relations.dict', header=None)
# #
# # train = original_id_to_triples_str(pos_triples, entity_to_id, rel_to_id)
# # test = original_id_to_triples_str(test_triples, entity_to_id, rel_to_id)
# # valid = original_id_to_triples_str(validation_triples, entity_to_id, rel_to_id)
# #
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/mapped/original_quadropoles.txt',np.array(train))
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/mapped/test.txt',np.array(test))
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/mapped/valid.txt',np.array(valid))
#
# #For kinship Datasets
# train_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/train.triples', dtype=str, comments='@Comment@ Subject Object Predicate')
# test_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/test.triples', dtype=str, comments='@Comment@ Subject Object Predicate')
# validation_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/dev.triples', dtype=str, comments='@Comment@ Subject Object Predicate')
# #validation_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/LogicENN/Data_Generator/DB100K/_test.txt', dtype=str, comments='@Comment@ Subject Predicate Object')
# #pos_triples = pos_triples[:,:-1]
# data_train = pd.DataFrame(train_triples, columns=["subject", "object", "predicate"])
# data_test = pd.DataFrame(test_triples, columns=["subject", "object", "predicate"])
# data_valid = pd.DataFrame(validation_triples, columns=["subject", "object", "predicate"])
# #data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "object"])
# data_train_ = data_train[['subject','predicate','object']]
# data_test_ = data_test[['subject','predicate','object']]
# data_valid_ = data_valid[['subject','predicate','object']]
# frames = [data_train_, data_test_, data_valid_]
# result = pd.concat(frames)
# result_in_array = np.array(result)
#
# #train_pos, test_pos = train_test_split(pos_triples, test_size=0.2)
# #train_pos, test_pos = train_test_split(pos_triples, test_size=0.2)
# #data = pd.DataFrame(pos_triples, columns=["subject", "predicate", "object"])
#
# entity_to_id, rel_to_id = create_mappings(triples=result_in_array)
# ent2id = dict((v,k) for k,v in entity_to_id.items())
# rel2id = dict((v,k) for k,v in rel_to_id.items())
#
# write_dic('/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/mapped/entities.dict',ent2id)
# write_dic('/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/mapped/relations.dict',rel2id)
#
# entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/mapped/entities.dict', header=None)
# rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/mapped/relations.dict', header=None)
#
# train = original_id_to_triples_str(data_train_, entity_to_id, rel_to_id)
# test = original_id_to_triples_str(data_test_, entity_to_id, rel_to_id)
# valid = original_id_to_triples_str(data_valid_, entity_to_id, rel_to_id)
# write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/mapped/original_quadropoles.txt',np.array(train))
# write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/mapped/test.txt',np.array(test))
# write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/Data_Generator/kinship/mapped/valid.txt',np.array(valid))
#
# mapped_triples = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/kinship_updated/mapped/original_quadropoles.txt', header=None)
# original_triples = mapped_id_to_original_triples(mapped_triples, entity_to_id, rel_to_id)
# write_to_txt_file('/home/mirza/PycharmProjects/baseModels_new_ver/kinship_updated/mapped/original_train.txt',np.array(original_triples))
#
# #For countries Datasets
# train_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/original_quadropoles.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# test_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/test.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# validation_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/valid.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
#
# data_train = pd.DataFrame(train_triples, columns=["subject", "object", "predicate"])
# data_test = pd.DataFrame(test_triples, columns=["subject", "object", "predicate"])
# data_valid = pd.DataFrame(validation_triples, columns=["subject", "object", "predicate"])
# #data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "object"])
# #data_train_ = data_train[['subject','predicate','object']]
# #data_test_ = data_test[['subject','predicate','object']]
# #data_valid_ = data_valid[['subject','predicate','object']]
# frames = [data_train, data_test, data_valid]
# result = pd.concat(frames)
# result_in_array = np.array(result)
#
# entity_to_id, rel_to_id = create_mappings(triples=result_in_array)
# ent2id = dict((v,k) for k,v in entity_to_id.items())
# rel2id = dict((v,k) for k,v in rel_to_id.items())
#
# write_dic('/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/mapped/entities.dict',ent2id)
# write_dic('/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/mapped/relations.dict',rel2id)
#
# entity_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/countries/entities.dict', header=None)
# rel_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/countries/relations.dict', header=None)
#
# train = original_id_to_triples_str(data_train, entity_to_id, rel_to_id)
# test = original_id_to_triples_str(data_test, entity_to_id, rel_to_id)
# valid = original_id_to_triples_str(data_valid, entity_to_id, rel_to_id)
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/mapped/original_quadropoles.txt',np.array(train))
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/mapped/test.txt',np.array(test))
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/countries_S1/mapped/valid.txt',np.array(valid))
#
#
# mapped_triples = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/countries/original_quadropoles.txt', header=None)
# original_triples = mapped_id_to_original_triples(mapped_triples, entity_to_id, rel_to_id)
# write_to_txt_file('/home/mirza/PycharmProjects/baseModels_new_ver/countries/original_train.txt',np.array(original_triples))
# #For nations dataset
#
# train_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/nations/original_quadropoles.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# test_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/nations/test.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# validation_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/nations/valid.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
#
# data_train = pd.DataFrame(train_triples, columns=["subject", "object", "predicate"])
# data_test = pd.DataFrame(test_triples, columns=["subject", "object", "predicate"])
# data_valid = pd.DataFrame(validation_triples, columns=["subject", "object", "predicate"])
# #data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "object"])
# #data_train_ = data_train[['subject','predicate','object']]
# #data_test_ = data_test[['subject','predicate','object']]
# #data_valid_ = data_valid[['subject','predicate','object']]
# frames = [data_train, data_test, data_valid]
# result = pd.concat(frames)
# result_in_array = np.array(result)
#
# entity_to_id, rel_to_id = create_mappings(triples=result_in_array)
# ent2id = dict((v,k) for k,v in entity_to_id.items())
# rel2id = dict((v,k) for k,v in rel_to_id.items())
#
# write_dic('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/entities.dict',ent2id)
# write_dic('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/relations.dict',rel2id)
#
# entity_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/nations_mapped/entities.dict', header=None)
# rel_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/nations_mapped/relations.dict', header=None)
#
# train = original_id_to_triples_str(data_train, entity_to_id, rel_to_id)
# test = original_id_to_triples_str(data_test, entity_to_id, rel_to_id)
# valid = original_id_to_triples_str(data_valid, entity_to_id, rel_to_id)
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/original_quadropoles.txt',np.array(train))
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/test.txt',np.array(test))
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/valid.txt',np.array(valid))
#
#
# mapped_triples = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/nations_mapped/original_quadropoles.txt', header=None)
# original_triples = mapped_id_to_original_triples(mapped_triples, entity_to_id, rel_to_id)
# write_to_txt_file('/home/mirza/PycharmProjects/baseModels_new_ver/nations_mapped/original_train.txt',np.array(original_triples))
#
#
# # #test = pd.read_table('/media/rony/Feel Free/Temp/Data/LogicENN/fb15k237/test.txt')
# # #train = pd.read_table('/media/rony/Feel Free/Temp/Data/LogicENN/fb15k237/original_quadropoles.txt')
# # #validation = pd.read_table('/media/rony/Feel Free/Temp/Data/LogicENN/fb15k237/valid.txt')
# #
# # #for dbpedia 2.0
# # mapped_pos_train_tripels = shuffle(mapped_pos_train_tripels)
# # train = mapped_pos_train_tripels[:27000,:]
# # test  = mapped_pos_train_tripels[27797 : 30000,:]
# # validation = mapped_pos_train_tripels[30000:32000,:]
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/test_dir/original_quadropoles.txt',train)
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/test_dir/test.txt',test)
# # write_to_txt_file('/home/mirza/PycharmProjects/LogicENN/test_dir/valid.txt',validation)
# #
# # ###########FOR CREATING MAPPING#############################################
# # ############For DB_WD_mapping_data##########################################
# # #For triple set 1 out of 2:
# #
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/mapping/triples_1.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/Data_Generator/WN18/WN18/entities.dict', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/Data_Generator/WN18/WN18/relations.dict', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/Data_Generator/mapped_data_wn_18/triples_2_original.tsv',index=False,sep='\t', header=None)
# #
# # #For triple set 2 out of 2:
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/mapping/db_wd_data/triples_2.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/mapping/ent_ids_2.txt', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/mapping/rel_ids_2.txt', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/LogicENN/db_wd_data/mapping/triples_2_original.tsv',index=False,sep='\t', header=None)
# #
# # #For DB_WD_sharing_data
# # #For triple set 1 out of 2:
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/triples_1.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/ent_ids_1.txt', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/rel_ids_1.txt', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/triples_1_original.tsv',index=False,sep='\t', header=None)
# #
# # #For triple set 2 out of 2:
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/triples_2.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/ent_ids_2.txt', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/rel_ids_2.txt', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/LogicENN/db_wd_data/sharing/triples_2_original.tsv',index=False,sep='\t', header=None)
# #
# # ###########FOR CREATING MAPPING#############################################
# # ############For DB_YG_mapping_data##########################################
# # #For triple set 1 out of 2:
# #
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/triples_1.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/ent_ids_1.txt', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/rel_ids_1.txt', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/triples_1_original.tsv',index=False,sep='\t', header=None)
# #
# # #For triple set 2 out of 2:
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/triples_2.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/ent_ids_2.txt', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/rel_ids_2.txt', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/LogicENN/db_yg_data/mapping/triples_2_original.tsv',index=False,sep='\t', header=None)
# #
# # #For DB_YG_sharing_data
# # #For triple set 1 out of 2:
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/triples_1.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/ent_ids_1.txt', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/rel_ids_1.txt', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/triples_1_original.tsv',index=False,sep='\t', header=None)
# #
# # #For triple set 2 out of 2:
# # mapped_pos_train_tripels = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/triples_2.txt', header=None)
# # entity_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/ent_ids_2.txt', header=None)
# # rel_to_id = pd.read_table('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/rel_ids_2.txt', header=None)
# #
# # original_triples = create_mapped_id_to_triples(mapped_pos_train_tripels, entity_to_id, rel_to_id)
# # original_triples.to_csv('/home/mirza/PycharmProjects/LogicENN/db_yg_data/sharing/triples_2_original.tsv',index=False,sep='\t', header=None)
# # For wordnet dataset
# train_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/original_quadropoles.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# test_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/test.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# validation_triples = np.loadtxt(fname='/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/valid.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
#
# data_train = pd.DataFrame(train_triples, columns=["subject", "predicate", "object"])
# data_test = pd.DataFrame(test_triples, columns=["subject", "predicate", "object"])
# data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "predicate"])
# #data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "object"])
# #data_train_ = data_train[['subject','predicate','object']]
# #data_test_ = data_test[['subject','predicate','object']]
# #data_valid_ = data_valid[['subject','predicate','object']]
# frames = [data_train, data_test, data_valid]
# result = pd.concat(frames)
# result_in_array = np.array(result)
#
# entity_to_id, rel_to_id = create_mappings(triples=result_in_array)
# ent2id = dict((v,k) for k,v in entity_to_id.items())
# rel2id = dict((v,k) for k,v in rel_to_id.items())
#
# write_dic('/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/entities.dict',ent2id)
# write_dic('/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/relations.dict',rel2id)
#
# entity_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/entities.dict', header=None)
# rel_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/relations.dict', header=None)
#
# train = original_id_to_triples(data_train, entity_to_id, rel_to_id)
# test = original_id_to_triples(data_test, entity_to_id, rel_to_id)
# valid = original_id_to_triples(data_valid, entity_to_id, rel_to_id)
# write_to_txt_file('/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/original_quadropoles.txt',np.array(train))
# write_to_txt_file('/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/test.txt',np.array(test))
# write_to_txt_file('/home/mirza/PycharmProjects/baseModels_new_ver/WN18RR/valid.txt',np.array(valid))
#
#
# mapped_triples = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/countries/original_quadropoles.txt', header=None)
# original_triples = mapped_id_to_original_triples(mapped_triples, entity_to_id, rel_to_id)
# write_to_txt_file('/home/mirza/PycharmProjects/baseModels_new_ver/countries/original_train.txt',np.array(original_triples))
# #For nations dataset
#
# train_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/nations/original_quadropoles.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# test_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/nations/test.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
# validation_triples = np.loadtxt(fname='/home/turzo/PycharmProjects/baseModels_new_ver/nations/valid.txt', dtype=str, comments='@Comment@ Subject Object Predicate')
#
# data_train = pd.DataFrame(train_triples, columns=["subject", "object", "predicate"])
# data_test = pd.DataFrame(test_triples, columns=["subject", "object", "predicate"])
# data_valid = pd.DataFrame(validation_triples, columns=["subject", "object", "predicate"])
# #data_valid = pd.DataFrame(validation_triples, columns=["subject", "predicate", "object"])
# #data_train_ = data_train[['subject','predicate','object']]
# #data_test_ = data_test[['subject','predicate','object']]
# #data_valid_ = data_valid[['subject','predicate','object']]
# frames = [data_train, data_test, data_valid]
# result = pd.concat(frames)
# result_in_array = np.array(result)
#
# entity_to_id, rel_to_id = create_mappings(triples=result_in_array)
# ent2id = dict((v,k) for k,v in entity_to_id.items())
# rel2id = dict((v,k) for k,v in rel_to_id.items())
#
# write_dic('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/entities.dict',ent2id)
# write_dic('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/relations.dict',rel2id)
#
# entity_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/nations_mapped/entities.dict', header=None)
# rel_to_id = pd.read_table('/home/mirza/PycharmProjects/baseModels_new_ver/nations_mapped/relations.dict', header=None)
#
# train = original_id_to_triples(data_train, entity_to_id, rel_to_id)
# test = original_id_to_triples(data_test, entity_to_id, rel_to_id)
# valid = original_id_to_triples(data_valid, entity_to_id, rel_to_id)
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/original_quadropoles.txt',np.array(train))
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/test.txt',np.array(test))
# write_to_txt_file('/home/turzo/PycharmProjects/baseModels_new_ver/nations/mapped/valid.txt',np.array(valid))