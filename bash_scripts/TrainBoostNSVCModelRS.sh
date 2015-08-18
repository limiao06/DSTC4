# remove stop words version
echo not finish yet
# the first model is not trained from sub_segments data, it is re-trained from an existing model's training samples
# TrainBoostNSVCModelRS.sh old_feature_list iteration_index new_feature_list iteration_nums
# ./TrainBoostNSVCModelRS.sh uB 3 u 5
set -u
set -e
# train boost model
# python ../scripts/dstc_thu/nsvc_boosting_method.py --dataset dstc4_train --dataroot ../data --subseg ../output/processed_data/sub_segments_data/sub_segments_train.json --old_model ../output/models/NSVC_models/nsvc_${1}_model_RS_boost/${2} ../output/models/NSVC_models/nsvc_newboost_${1}${2}_${3}_model_RS --it ${4} --ReTrain --RemoveSW --feature ${3} 
