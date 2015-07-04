python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.8.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.8 --exact
python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.7.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.7 --exact
python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.6.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.6 --exact
python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.5.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.5 --exact

python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.8.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.8
python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.7.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.7
python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.6.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.6
python ../scripts/dstc_thu/association_rule_tracker.py --dataset dstc4_dev --dataroot ../data/ --ar ../output/models/association_rule_models/association_train_rule.json --stm ../output/models/SemTagModel/semtag_train_model --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.5.json --ontology ../scripts/config/ontology_dstc4.json --pt 0.5

python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.8.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.8.score.csv --ontology ../scripts/config/ontology_dstc4.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.7.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.7.score.csv --ontology ../scripts/config/ontology_dstc4.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.6.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.6.score.csv --ontology ../scripts/config/ontology_dstc4.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.5.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.5.score.csv --ontology ../scripts/config/ontology_dstc4.json

python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.8.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.8.score.csv --ontology ../scripts/config/ontology_dstc4.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.7.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.7.score.csv --ontology ../scripts/config/ontology_dstc4.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.6.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.6.score.csv --ontology ../scripts/config/ontology_dstc4.json
python ../scripts/score.py --dataset dstc4_dev --dataroot ../data/ --trackfile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.5.json --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.5.score.csv --ontology ../scripts/config/ontology_dstc4.json

rm ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
echo ass_rule_exact_out_t0.8.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.8.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
echo ass_rule_exact_out_t0.7.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.7.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
echo ass_rule_exact_out_t0.6.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.6.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
echo ass_rule_exact_out_t0.5.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_exact_out_t0.5.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt

echo ass_rule_fuzzy_out_t0.8.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.8.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
echo ass_rule_fuzzy_out_t0.7.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.7.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
echo ass_rule_fuzzy_out_t0.6.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.6.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
echo ass_rule_fuzzy_out_t0.5.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
python ../scripts/report.py --scorefile ../output/msiip_out/ass_rule_out/ass_rule_fuzzy_out_t0.5.score.csv >> ../output/msiip_out/ass_rule_out/ass_rule_out_result.txt
