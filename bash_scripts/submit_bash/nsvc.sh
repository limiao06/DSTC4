# nsvc_u
# nsvc_u test or baseline dev
base_path=../..
if [ $# -ne 2 ];then
	echo ${0} "u|uB test|dev"
	exit
fi

if [ ${2} == dev ];then
	python ${base_path}/scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json \
		--model_dir ${base_path}/submit/nsvc/Feat_${1}/model/ \
    --slot_prob 0.5 \
		--trackfile ${base_path}/submit/nsvc/Feat_${1}/answer_dev.json
	python ${base_path}/scripts/score.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--trackfile ${base_path}/submit/nsvc/Feat_${1}/answer_dev.json \
		--scorefile ${base_path}/submit/nsvc/Feat_${1}/answer_dev.score \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
	python ${base_path}/scripts/report.py --scorefile ${base_path}/submit/nsvc/Feat_${1}/answer_dev.score
fi

if [ ${2} == "test" ];then
	python ${base_path}/scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_test --dataroot ${base_path}/data \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json \
		--model_dir ${base_path}/submit/nsvc/Feat_${1}/model/ \
		--trackfile ${base_path}/submit/nsvc/Feat_${1}/answer_test.json
fi
