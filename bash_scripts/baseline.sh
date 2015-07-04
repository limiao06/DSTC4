python ../scripts/baseline.py --dataset dstc4_dev --dataroot ../data --trackfile ../output/msiip_out/baseline_out/baseline_out.json --ontology ../scripts/config/ontology_dstc4.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data --trackfile ../output/msiip_out/baseline_out/baseline_out.json --ontology ../scripts/config/ontology_dstc4.json --scorefile ../output/msiip_out/baseline_out/baseline_out.score
rm ../output/msiip_out/baseline_out/baseline_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/baseline_out/baseline_out.score > ../output/msiip_out/baseline_out/baseline_out_result.txt
