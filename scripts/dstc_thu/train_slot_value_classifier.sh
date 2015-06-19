python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tM_c0.3_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.3 --mode MINE
python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tM_c0.5_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.5 --mode MINE
python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tM_c0.7_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.7 --mode MINE
python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tM_c0.9_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.9 --mode MINE

python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tN_c0.3_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.3 --mode NLTK
python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tN_c0.5_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.5 --mode NLTK
python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tN_c0.7_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.7 --mode NLTK
python Slot_value_classifier.py ../../output/sub_segments.json ../../output/slot_value_TuB_tN_c0.9_model --train --ontology ../config/ontology_dstc4.json --feature TuB --percent 0.9 --mode NLTK
