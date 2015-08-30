# baseline
# baseline test or baseline dev
set -e
set -u
base_path=../..
if [ $# -ne 1 ];then
	echo ${0} "test | dev"
	exit
fi

if [ ${1} == dev ];then
	 python ${base_path}/scripts/dstc_thu/msiip_new_ensemble_tracker.py --dataset dstc4_dev --dataroot ${base_path}/data/ \
     --LogBaseDir ${base_path}/submit/ --config ${base_path}/submit/ensemble/config_dev \
     --trackfile ${base_path}/submit/ensemble/answer_dev.json \
     --ontology ${base_path}/scripts/config/ontology_dstc4.json \
#     --weight_key precision
	python ${base_path}/scripts/score.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--trackfile ${base_path}/submit/ensemble/answer_dev.json \
		--scorefile ${base_path}/submit/ensemble/answer_dev.score \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
	python ${base_path}/scripts/report.py --scorefile ${base_path}/submit/ensemble/answer_dev.score
fi

if [ ${1} == "test" ];then
	 python ${base_path}/scripts/dstc_thu/msiip_new_ensemble_tracker.py --dataset dstc4_test --dataroot ${base_path}/data/ \
     --LogBaseDir ${base_path}/submit/ --config ${base_path}/submit/ensemble/config_test \
     --trackfile ${base_path}/submit/ensemble/answer_test.json \
     --ontology ${base_path}/scripts/config/ontology_dstc4.json
#     --weight_key precision
fi
