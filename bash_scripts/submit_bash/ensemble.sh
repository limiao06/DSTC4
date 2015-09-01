# baseline
# baseline test or baseline dev
set -e
set -u
base_path=../..
if [ $# -ne 3 ];then
	echo ${0} "1|2 test | dev thres"
	exit
fi

if [ ${2} == dev ];then
	 python ${base_path}/scripts/dstc_thu/msiip_new_ensemble_tracker.py --dataset dstc4_dev --dataroot ${base_path}/data/ \
     --LogBaseDir ${base_path}/submit/ --config ${base_path}/submit/ensemble/config_${1}_dev \
     --trackfile ${base_path}/submit/ensemble/answer_${1}_dev.json \
     --ontology ${base_path}/scripts/config/ontology_dstc4.json \
     --unified_thres ${3}
	python ${base_path}/scripts/score.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--trackfile ${base_path}/submit/ensemble/answer_${1}_dev.json \
		--scorefile ${base_path}/submit/ensemble/answer_${1}_dev.score \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
	python ${base_path}/scripts/report.py --scorefile ${base_path}/submit/ensemble/answer_${1}_dev.score
fi

if [ ${2} == "test" ];then
	 python ${base_path}/scripts/dstc_thu/msiip_new_ensemble_tracker.py --dataset dstc4_test --dataroot ${base_path}/data/ \
     --LogBaseDir ${base_path}/submit/ --config ${base_path}/submit/ensemble/config_${1}_test \
     --trackfile ${base_path}/submit/ensemble/answer_${1}_test.json \
     --ontology ${base_path}/scripts/config/ontology_dstc4.json \
     --unified_thres ${3}
fi
