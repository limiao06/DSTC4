# baseline
# baseline test or baseline dev
base_path=../..
logfile=${base_path}/submit/ensemble/scan_unified_thres.txt
thres_vec=(0.35 0.375 0.4 0.425 0.45)
echo "scan unified_thres:" > ${logfile}
for thres in ${thres_vec[@]}
do
	 python ${base_path}/scripts/dstc_thu/msiip_new_ensemble_tracker.py --dataset dstc4_dev --dataroot ${base_path}/data/ \
     --LogBaseDir ${base_path}/submit/ --config ${base_path}/submit/ensemble/config_1_dev \
     --trackfile ${base_path}/submit/ensemble/answer_dev_$thres.json \
     --ontology ${base_path}/scripts/config/ontology_dstc4.json \
     --unified_thres $thres
#     --weight_key precision
	python ${base_path}/scripts/score.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--trackfile ${base_path}/submit/ensemble/answer_dev_$thres.json \
		--scorefile ${base_path}/submit/ensemble/answer_dev_$thres.score \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
	echo "unified thres: " $thres >> ${logfile}
  python ${base_path}/scripts/report.py --scorefile ${base_path}/submit/ensemble/answer_dev_$thres.score >> ${logfile}
done
