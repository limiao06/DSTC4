rm ../../output/msiip_out/msiip_nsvc_out/test_param/test_slot_prob_thres.txt
set -e
for t in $(seq 0.5 0.2 0.9)
do
  python ../../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../../data/ --model_dir ../../output/models/NSVC_models/nsvc_uB_model_boost/3/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_slot_prob_t${t}_hr_08.json --ontology ../../scripts/config/ontology_dstc4.json --slot_prob ${t}
  python ../../scripts/score.py --dataset dstc4_dev --dataroot ../../data/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_slot_prob_t${t}_hr_08.json --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_slot_prob_t${t}_hr_08.score --ontology ../../scripts/config/ontology_dstc4.json
  echo 'slot_prob_threshold' ${t} >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_slot_prob_thres.txt
  python ../../scripts/report.py --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_slot_prob_t${t}_hr_08.score >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_slot_prob_thres.txt
done
