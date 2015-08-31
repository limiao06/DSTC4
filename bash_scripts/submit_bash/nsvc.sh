# nsvc_u
# nsvc_u test or baseline dev
base_path=../..
if [ $# -ne 3 ];then
	echo ${0} "u|uB 1 test|dev"
	exit
fi

if [ ${3} == dev ];then
	python ${base_path}/scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json \
		--model_dir ${base_path}/submit/nsvc/Feat_${1}/model${2}/ \
    --slot_prob 0.5 \
		--trackfile ${base_path}/submit/nsvc/Feat_${1}/answer_${2}_dev.json
	python ${base_path}/scripts/score.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--trackfile ${base_path}/submit/nsvc/Feat_${1}/answer_${2}_dev.json \
		--scorefile ${base_path}/submit/nsvc/Feat_${1}/answer_${2}_dev.score \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
	python ${base_path}/scripts/report.py --scorefile ${base_path}/submit/nsvc/Feat_${1}/answer_${2}_dev.score
fi

if [ ${3} == "test" ];then
	python ${base_path}/scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_test --dataroot ${base_path}/data \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json \
		--model_dir ${base_path}/submit/nsvc/Feat_${1}/model${2}/ \
		--trackfile ${base_path}/submit/nsvc/Feat_${1}/answer_${2}_test.json
fi
