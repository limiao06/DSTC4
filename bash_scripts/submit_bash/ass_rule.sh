# ass_rule
# ass_rule test or baseline dev
base_path=../..
if [ $# -ne 1 ];then
	echo ${0} "test | dev"
	exit
fi

if [ ${1} == dev ];then
	python ${base_path}/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--ar ${base_path}/output/models/association_rule_models/association_train_rule.json
		--stm ${base_path}/output/models/SemTagModel/semtag_train_model
		--trackfile ${base_path}/submit/ass_rule/answer_dev.json \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
		--pt 0.8
	python ${base_path}/scripts/score.py --dataset dstc4_dev --dataroot ${base_path}/data \
		--trackfile ${base_path}/submit/ass_rule/answer_dev.json \
		--scorefile ${base_path}/submit/ass_rule/answer_dev.score \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
	python ${base_path}/scripts/report.py --scorefile ${base_path}/submit/ass_rule/answer_dev.score
fi

if [ ${1} == "test" ];then
	python ${base_path}/dstc_thu/association_rule_tracker.py --dataset dstc4_test --dataroot ${base_path}/data \
		--ar ${base_path}/output/models/association_rule_models/association_train_rule.json
		--stm ${base_path}/output/models/SemTagModel/semtag_train_model
		--trackfile ${base_path}/submit/ass_rule/answer_test.json \
		--ontology ${base_path}/scripts/config/ontology_dstc4.json
		--pt 0.8
fi
