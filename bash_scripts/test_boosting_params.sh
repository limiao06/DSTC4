# test_boosting_params.sh feature_list iteration_times
# ./test_boosting_params.sh uB 5
# train base model

set -u
set -e
echo "test_boosting_params" ${1} > ../output/msiip_out/msiip_nsvc_out/test_boosting_params/test_boosting_params.txt

python ../scripts/dstc_thu/nsvc_sub_segments_tools.py --subseg ../output/processed_data/sub_segments_data/sub_segments_train.json --train --ontology ../scripts/config/ontology_dstc4.json --feature ${1} ../output/models/NSVC_models/nsvc_${1}_model
python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data --ontology ../scripts/config/ontology_dstc4.json --model_dir ../output/models/NSVC_models/nsvc_${1}_model/ --trackfile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_t80_hr.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_t80_hr.json --scorefile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_t80_hr.score --ontology ../scripts/config/ontology_dstc4.json
echo "base" > ../output/msiip_out/msiip_nsvc_out/test_boosting_params/test_boosting_params.txt
python ../scripts/report.py --scorefile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_t80_hr.score >> ../output/msiip_out/msiip_nsvc_out/test_boosting_params/test_boosting_params.txt


high_thres_vec=(0.6 0.7 0.75 0.8 0.85 0.9)
low_thres_vec=(0.1 0.2 0.3 0.4)

for high_thres in ${high_thres_vec[@]}
do
	for low_thres in ${low_thres_vec[@]}
	do
		python ../scripts/dstc_thu/nsvc_boosting_method.py --dataset dstc4_train --dataroot ../data --subseg ../output/processed_data/sub_segments_data/sub_segments_train.json --old_model ../output/models/NSVC_models/nsvc_${1}_model ../output/models/NSVC_models/nsvc_test_boost/nsvc_boost_${1}_HT${high_thres}_LT{low_thres} --it ${2} --ht ${high_thres} -- lt ${low_thres}
		for it in $(seq 0 $[${2}-1])
		do
		  python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data/ --model_dir ../output/models/NSVC_models/nsvc_test_boost/nsvc_boost_${1}_HT${high_thres}_LT{low_thres}/${it}/ --trackfile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_boost_HT${high_thres}_LT{low_thres}_${it}_t80_hr.json --ontology ../scripts/config/ontology_dstc4.json
		  python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_boost_HT${high_thres}_LT{low_thres}_${it}_t80_hr.json --scorefile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_boost_HT${high_thres}_LT{low_thres}_${it}_t80_hr.score --ontology ../scripts/config/ontology_dstc4.json
		  echo "high_thres:" $high_thres ", low_thres:" $low_thres ", boost" ${it} >> ../output/msiip_out/msiip_nsvc_out/test_boosting_params/test_boosting_params.txt
		  python ../scripts/report.py --scorefile ../output/msiip_out/msiip_nsvc_out/test_boosting_params/msiip_nsvc_${1}_boost_HT${high_thres}_LT{low_thres}_${it}_t80_hr.score >> ../output/msiip_out/msiip_nsvc_out/test_boosting_params/test_boosting_params.txt
		done
	done
done




