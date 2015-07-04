python ../my_code/Orange_File_Builder.py ../output/processed_data/sub_segments_data/sub_segments_train.json ../output/processed_data/Orange_files/Orang_train_files
python ../my_code/Orange_File_Builder.py ../output/processed_data/sub_segments_data/sub_segments_train.json ../output/processed_data/Orange_files/Orang_train_files -v
python ../my_code/Orange_File_Builder.py ../output/processed_data/sub_segments_data/sub_segments_train.json ../output/processed_data/Orange_files/Orang_train_files -v -s

python ../my_code/Test_Orange_AssociationRules.py ../output/processed_data/Orange_files/Orang_train_files/ ../output/models/association_rule_models/association_train_rules_s2 -s 2

python ../my_code/association_rules_reader.py ../output/models/association_rule_models/association_train_rules_s2 ../output/models/association_rule_models/association_train_rule.json

