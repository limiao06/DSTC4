# remove stop words version
# TrainNSVCModel.sh feature_list iteration_times
# ./TrainNSVCModel.sh uB 5
# train base model
set -u
set -e
python ../scripts/dstc_thu/nsvc_sub_segments_tools.py --subseg ../output/processed_data/sub_segments_data/sub_segments_train.json --train --ontology ../scripts/config/ontology_dstc4.json --feature ${1} ../output/models/NSVC_models/nsvc_${1}_model_RS --RemoveSW

# train boost model
python ../scripts/dstc_thu/nsvc_boosting_method.py --dataset dstc4_train --dataroot ../data --subseg ../output/processed_data/sub_segments_data/sub_segments_train.json --old_model ../output/models/NSVC_models/nsvc_${1}_model_RS ../output/models/NSVC_models/nsvc_${1}_model_RS_boost --it ${2}
