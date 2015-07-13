rm ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode.txt
set -e
python ../../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../../data/ --model_dir ../../output/models/NSVC_models/nsvc_uB_model_boost/3/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_max_08.json --ontology ../../scripts/config/ontology_dstc4.json --BSMode max
python ../../scripts/score.py --dataset dstc4_dev --dataroot ../../data/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_max_08.json --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_max_08.score --ontology ../../scripts/config/ontology_dstc4.json
echo 'bs mode max' >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode.txt
python ../../scripts/report.py --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_max_08.score >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode.txt

python ../../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../../data/ --model_dir ../../output/models/NSVC_models/nsvc_uB_model_boost/3/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_enhance_08.json --ontology ../../scripts/config/ontology_dstc4.json --BSMode enhance
python ../../scripts/score.py --dataset dstc4_dev --dataroot ../../data/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_enhance_08.json --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_enhance_08.score --ontology ../../scripts/config/ontology_dstc4.json
echo 'bs mode enhance' >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode.txt
python ../../scripts/report.py --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_enhance_08.score >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode.txt

for alpha in $(seq 0.0 0.2 0.6)
do
  python ../../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../../data/ --model_dir ../../output/models/NSVC_models/nsvc_uB_model_boost/3/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_average_a${alpha}_08.json --ontology ../../scripts/config/ontology_dstc4.json --BSMode average --BSAlpha ${alpha}
  python ../../scripts/score.py --dataset dstc4_dev --dataroot ../../data/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_average_a${alpha}_08.json --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_average_a${alpha}_08.score --ontology ../../scripts/config/ontology_dstc4.json
  echo 'bs mode average, alpha' ${alpha} >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode.txt
  python ../../scripts/report.py --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode_hr_average_a${alpha}_08.score >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_bs_mode.txt
done
