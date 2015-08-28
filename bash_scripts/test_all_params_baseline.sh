outdir=../output/msiip_out/baseline_out/test_all_params
logfile=${outdir}/test_all_params.txt
echo 'test_all_params:' > ${logfile}
set -e

ratio_thres_vec=(0.75 0.8 0.85 0.9)
bs_alpha_vec=(0.0 0.1 0.2 0.3)
bs_mode_vec=("max" "enhance")

for ratio_thres in ${ratio_thres_vec[@]}
do
	for bs_mode in ${bs_mode_vec[@]}
	do
		echo "ratio_thres:" $ratio_thres ", bs_mode:" $bs_mode >> ${logfile}
		python ../scripts/dstc_thu/msiip_baseline_tracker.py --dataset dstc4_dev --dataroot ../data/ \
			--ontology ../scripts/config/ontology_dstc4.json \
			--trackfile ${outdir}/test_all_params_RT${ratio_thres}_BM${bs_mode}.json \
			--ratio_thres ${ratio_thres} --BSMode ${bs_mode}
		python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ \
			--ontology ../scripts/config/ontology_dstc4.json \
			--trackfile ${outdir}/test_all_params_RT${ratio_thres}_BM${bs_mode}.json \
			--scorefile ${outdir}/test_all_params_RT${ratio_thres}_BM${bs_mode}.score
		python ../scripts/report.py \
			--scorefile ${outdir}/test_all_params_RT${ratio_thres}_BM${bs_mode}.score >> ${logfile}
	done

	for bs_alpha in ${bs_alpha_vec[@]}
	do
		echo "ratio_thres:" $ratio_thres ", bs_mode: average, bs_alpha:" $bs_alpha >> ${logfile}
		python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data/ \
			--model_dir ../output/models/NSVC_models/nsvc_uB_model_boost/3/ \
			--ontology ../scripts/config/ontology_dstc4.json \
			--trackfile ${outdir}/test_all_params_RT${ratio_thres}_BMaver_BA${bs_alpha}.json \
			--ratio_thres ${ratio_thres} --BSMode average --BSAlpha ${bs_alpha}
		python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ \
			--ontology ../scripts/config/ontology_dstc4.json \
			--trackfile ${outdir}/test_all_params_RT${ratio_thres}_BMaver_BA${bs_alpha}.json \
			--scorefile ${outdir}/test_all_params_RT${ratio_thres}_BMaver_BA${bs_alpha}.score
		python ../scripts/report.py \
			--scorefile ${outdir}/test_all_params_RT${ratio_thres}_BMaver_BA${bs_alpha}.score >> ${logfile}
	done			
done




