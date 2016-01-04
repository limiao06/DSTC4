log=../../output/msiip_out/msiip_nsvc_out/test_param/test_classif_prob_thres.txt
rm $log
model_dir=../../submit/nsvc/Feat_uB/model1
set -e
for t in $(seq 0.3 0.35 0.4 0.45 0.5 0.6)
do
  python ../../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../../data/ \
  	--model_dir ${model_dir} --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_classif_prob_t${t}_hr_08.json \
  	--ontology ../../scripts/config/ontology_dstc4.json --classif_prob ${t}
  python ../../scripts/score.py --dataset dstc4_dev --dataroot ../../data/ \
  	--trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_classif_prob_t${t}_hr_08.json \
  	--scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_classif_prob_t${t}_hr_08.score \
  	--ontology ../../scripts/config/ontology_dstc4.json
  echo 'slot_prob_threshold' ${t} >> $log
  python ../../scripts/report.py --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_classif_prob_t${t}_hr_08.score >> $log
done
