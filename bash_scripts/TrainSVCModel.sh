for i in $(seq 1 2 7)
do
  python ../scripts/dstc_thu/Slot_value_classifier.py --train --ontology ../scripts/config/ontology_dstc4.json ../output/processed_data/sub_segments_data/sub_segments_train.json ../output/models/SVC_models/svc_TuB_t0${i}_model --percent 0.${i} --feature TuB
done

