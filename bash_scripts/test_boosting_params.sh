# test_boosting_params.sh feature_list iteration_times
# ./test_boosting_params.sh uB 5
# train base model
set -u
set -e

train=1
if [ $# -eq 3 ];then
	if [ ${3} == NT ];then
		train=0
	fi
fi
if [ ${train} -eq 1 ];then
	echo train model
else
	echo no-train model
fi

logfile=../output/msiip_out/msiip_nsvc_out/test_boosting_params/test_boosting_params_${1}_${2}.txt
outfile_path=../output/msiip_out/msiip_nsvc_out/test_boosting_params
outmodel_path=../output/models/NSVC_models/nsvc_test_boost
echo "test_boosting_params" ${1} > ${logfile}
if [ $train -eq 1 ];then
	python ../scripts/dstc_thu/nsvc_sub_segments_tools.py \
		--subseg ../output/processed_data/sub_segments_data/sub_segments_train.json \
		--train --ontology ../scripts/config/ontology_dstc4.json --feature ${1} \
		../output/models/NSVC_models/nsvc_${1}_model
fi
python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data \
	--ontology ../scripts/config/ontology_dstc4.json \
	--model_dir ../output/models/NSVC_models/nsvc_${1}_model/ \
	--trackfile ${outfile_path}/msiip_nsvc_${1}_t80_hr.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data \
	--trackfile ${outfile_path}/msiip_nsvc_${1}_t80_hr.json \
	--scorefile ${outfile_path}/msiip_nsvc_${1}_t80_hr.score \
	--ontology ../scripts/config/ontology_dstc4.json
echo "base" > ${logfile}
python ../scripts/report.py --scorefile ${outfile_path}/msiip_nsvc_${1}_t80_hr.score >> ${logfile}


high_thres_vec=(0.6 0.7 0.8 0.9)
low_thres_vec=(0.1 0.2 0.3 0.4)

for high_thres in ${high_thres_vec[@]}
do
	for low_thres in ${low_thres_vec[@]}
	do
		if [ $train -eq 1 ];then
			python ../scripts/dstc_thu/nsvc_boosting_method.py --dataset dstc4_train --dataroot ../data \
				--subseg ../output/processed_data/sub_segments_data/sub_segments_train.json \
				--old_model ../output/models/NSVC_models/nsvc_${1}_model \
				${outmodel_path}/nsvc_boost_${1}_HT${high_thres}_LT${low_thres} \
				--it ${2} --ht ${high_thres} --lt ${low_thres}
		fi
		for it in $(seq 0 $[${2}-1])
		do
			python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data/ \
				--model_dir ${outmodel_path}/nsvc_boost_${1}_HT${high_thres}_LT${low_thres}/${it}/ \
				--trackfile ${outfile_path}/msiip_nsvc_${1}_boost_HT${high_thres}_LT${low_thres}_${it}_t80_hr.json \
				--ontology ../scripts/config/ontology_dstc4.json
			python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ \
				--trackfile ${outfile_path}/msiip_nsvc_${1}_boost_HT${high_thres}_LT${low_thres}_${it}_t80_hr.json \
				--scorefile ${outfile_path}/msiip_nsvc_${1}_boost_HT${high_thres}_LT${low_thres}_${it}_t80_hr.score \
				--ontology ../scripts/config/ontology_dstc4.json
			echo "high_thres:" $high_thres ", low_thres:" $low_thres ", boost" ${it} >> ${logfile}
			python ../scripts/report.py --scorefile ${outfile_path}/msiip_nsvc_${1}_boost_HT${high_thres}_LT${low_thres}_${it}_t80_hr.score >> ${logfile}
		done
	done
done




