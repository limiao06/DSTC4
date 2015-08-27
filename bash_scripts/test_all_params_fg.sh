logfile=../output/msiip_out/msiip_nsvc_out/test_all_params/test_all_params_fg.txt
echo 'test_all_params:' > ${logfile}
set -e

slot_prob_vec=(0.55 0.6 0.65)
value_prob_vec=(0.75 0.8 0.85)
stc_mode_vec=("hr" "hp")
bs_mode=enhance

for slot_prob in ${slot_prob_vec[@]}
do
	for value_prob in ${value_prob_vec[@]}
	do
		for stc_mode in ${stc_mode_vec[@]}
		do
			echo "slot_prob:" $slot_prob ", value_prob:" $value_prob ", stc_mode:" $stc_mode ", bs_mode:" $bs_mode >> ${logfile}
			python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data/ --model_dir ../output/models/NSVC_models/nsvc_uB_model_boost/3/ --ontology ../scripts/config/ontology_dstc4.json --trackfile ../output/msiip_out/msiip_nsvc_out/test_all_params/test_all_params_SP${slot_prob}_VP${value_prob}_SM${stc_mode}_BM${bs_mode}_t08.json --value_prob ${value_prob} --slot_prob ${slot_prob} --STCMode ${stc_mode} --BSMode ${bs_mode}
			python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --ontology ../scripts/config/ontology_dstc4.json --trackfile ../output/msiip_out/msiip_nsvc_out/test_all_params/test_all_params_SP${slot_prob}_VP${value_prob}_SM${stc_mode}_BM${bs_mode}_t08.json --scorefile ../output/msiip_out/msiip_nsvc_out/test_all_params/test_all_params_SP${slot_prob}_VP${value_prob}_SM${stc_mode}_BM${bs_mode}_t08.score
			python ../scripts/report.py --scorefile ../output/msiip_out/msiip_nsvc_out/test_all_params/test_all_params_SP${slot_prob}_VP${value_prob}_SM${stc_mode}_BM${bs_mode}_t08.score >> ${logfile}		
		done
	done
done
