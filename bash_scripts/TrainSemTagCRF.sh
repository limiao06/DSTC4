# generate crf data
echo generate crf data
python ../scripts/dstc_thu/Semantic_Tag_Data_Extractor.py ../output/processed_data/sub_utter_data/sub_utter_train.json ../output/processed_data/SemTagCRFData/SemTag_train.crf
python ../scripts/dstc_thu/Semantic_Tag_Data_Extractor.py ../output/processed_data/sub_utter_data/sub_utter_dev.json ../output/processed_data/SemTagCRFData/SemTag_dev.crf
# train crf
echo train crf
crf_learn -c 4.0 ../output/processed_data/SemTagCRFData/template ../output/processed_data/SemTagCRFData/SemTag_train.crf ../output/models/SemTagModel/semtag_train_model
# test crf in dev set
echo test crf in dev set
crf_test -m ../output/models/SemTagModel/semtag_train_model ../output/processed_data/SemTagCRFData/SemTag_dev.crf > ../output/processed_data/SemTagCRFData/SemTag_dev.crf.out
# eval crf in dev set
echo eval crf in dev set
python ../my_code/Eval_CRF.py ../output/processed_data/SemTagCRFData/SemTag_dev.crf.out
