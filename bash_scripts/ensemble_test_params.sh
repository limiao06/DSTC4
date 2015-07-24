echo 'ensemble test_params:' > ../output/msiip_out/msiip_ensemble_out/ensemble_test_params.txt
set -e

slot_prob_vec=(0.5 0.6 0.7)
value_prob_vec=(0.5 0.6 0.7 0.8 0.9)

for slot_prob in ${slot_prob_vec[@]}
do
	for value_prob in ${value_prob_vec[@]}
	do
		python ../scripts/dstc_thu/msiip_ensemble_tracker.py --LogBaseDir ../output/msiip_out/ --config ../scripts/config/ensemble_config.cfg --ontology ../scripts/config/ontology_dstc4.json --trackfile ../output/msiip_out/msiip_ensemble_out/msiip_ensemble_SP${slot_prob}_VP${value_prob}.json --dataset dstc4_dev --dataroot ../data/ --value_prob ${value_prob} --slot_prob ${slot_prob}
		python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --ontology ../scripts/config/ontology_dstc4.json --trackfile ../output/msiip_out/msiip_ensemble_out/msiip_ensemble_SP${slot_prob}_VP${value_prob}.json --scorefile ../output/msiip_out/msiip_ensemble_out/msiip_ensemble_SP${slot_prob}_VP${value_prob}.score
		python ../scripts/report.py --scorefile ../output/msiip_out/msiip_ensemble_out/msiip_ensemble_SP${slot_prob}_VP${value_prob}.score >> ../output/msiip_out/msiip_ensemble_out/ensemble_test_params.txt
	done
done




