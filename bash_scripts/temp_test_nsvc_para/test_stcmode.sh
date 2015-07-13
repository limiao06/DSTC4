rm ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode.txt
set -e
python ../../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../../data/ --model_dir ../../output/models/NSVC_models/nsvc_uB_model_boost/3/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hr_08.json --ontology ../../scripts/config/ontology_dstc4.json --STCMode hr
python ../../scripts/score.py --dataset dstc4_dev --dataroot ../../data/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hr_08.json --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hr_08.score --ontology ../../scripts/config/ontology_dstc4.json
echo 'stc mode high recall' >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode.txt
python ../../scripts/report.py --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hr_08.score >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode.txt

python ../../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../../data/ --model_dir ../../output/models/NSVC_models/nsvc_uB_model_boost/3/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hp_08.json --ontology ../../scripts/config/ontology_dstc4.json --STCMode hp
python ../../scripts/score.py --dataset dstc4_dev --dataroot ../../data/ --trackfile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hp_08.json --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hp_08.score --ontology ../../scripts/config/ontology_dstc4.json
echo 'stc mode high precision' >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode.txt
python ../../scripts/report.py --scorefile ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode_hp_08.score >> ../../output/msiip_out/msiip_nsvc_out/test_param/test_stc_mode.txt
