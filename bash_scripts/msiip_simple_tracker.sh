rm ../output/msiip_out/msiip_simple_out/msiip_simple_result.txt
for i in $(seq 1 2 7)
do
  python ../scripts/dstc_thu/msiip_simple_tracker.py --dataset dstc4_dev --dataroot ../data --ontology ../scripts/config/ontology_dstc4.json --model_dir ../output/models/SVC_models/svc_TuB_t0${i}_model/ --trackfile ../output/msiip_out/msiip_simple_out/msiip_simple_TuB_t0${i}.json
  python ../scripts/score.py --dataset dstc4_dev --dataroot ../data --ontology ../scripts/config/ontology_dstc4.json --trackfile ../output/msiip_out/msiip_simple_out/msiip_simple_TuB_t0${i}.json --scorefile ../output/msiip_out/msiip_simple_out/msiip_simple_TuB_t0${i}.score
  echo msiip_simple_TuB_t0${i}.score >> ../output/msiip_out/msiip_simple_out/msiip_simple_result.txt
  python ../scripts/report.py --scorefile ../output/msiip_out/msiip_simple_out/msiip_simple_TuB_t0${i}.score >> ../output/msiip_out/msiip_simple_out/msiip_simple_result.txt
done
