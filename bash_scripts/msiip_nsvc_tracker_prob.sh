# msiip_nsvc_tracker.sh feature iteration
# msiip_nsvc_tracker.sh uB 5
set -e
set -u
logfile=../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_prob_result.txt
rm $logfile
python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data --ontology ../scripts/config/ontology_dstc4.json --model_dir ../output/models/NSVC_models/nsvc_${1}_model/ --trackfile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_t80_hr.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_t80_hr.json --scorefile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_t80_hr.score --ontology ../scripts/config/ontology_dstc4.json
echo ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_t80_hr.score > $logfile
python ../scripts/report.py --scorefile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_t80_hr.score >> $logfile


for it in $(seq 0 $[${2}-1])
do
  python ../scripts/dstc_thu/msiip_nsvc_tracker.py --dataset dstc4_dev --dataroot ../data/ --model_dir ../output/models/NSVC_models/nsvc_${1}_model_prob_boost/${it}/ --trackfile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_prob_boost${it}_t80_hr.json --ontology ../scripts/config/ontology_dstc4.json
  python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_prob_boost${it}_t80_hr.json --scorefile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_prob_boost${it}_t80_hr.score --ontology ../scripts/config/ontology_dstc4.json
  echo 'boost' ${it} >> $logfile
  python ../scripts/report.py --scorefile ../output/msiip_out/msiip_nsvc_out/msiip_nsvc_${1}_prob_boost${it}_t80_hr.score >> $logfile
done

