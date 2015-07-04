# extract sub_segments files
python ../my_code/temp_extract_sub_segments.py --dataset dstc4_train --dataroot ../data --trackfile ../output/processed_data/sub_segments_data/sub_segments_train.json --ontology ../scripts/config/ontology_dstc4.json
python ../my_code/temp_extract_sub_segments.py --dataset dstc4_dev --dataroot ../data --trackfile ../output/processed_data/sub_segments_data/sub_segments_dev.json --ontology ../scripts/config/ontology_dstc4.json
