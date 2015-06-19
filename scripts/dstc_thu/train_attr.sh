#python Attributes_Classifier.py ../../processed_data/attr_data_train.json ../../output/attr_models/attr_model_u08 --percent 0.8 --train --feature u
#python Attributes_Classifier.py ../../processed_data/attr_data_train.json ../../output/attr_models/attr_model_u05 --percent 0.5 --train --feature u
#python Attributes_Classifier.py ../../processed_data/attr_data_train.json ../../output/attr_models/attr_model_u03 --percent 0.3 --train --feature u


#python Attributes_Classifier.py ../../processed_data/attr_data_train.json ../../output/attr_models/attr_model_ub08 --percent 0.8 --train --feature ub
#python Attributes_Classifier.py ../../processed_data/attr_data_train.json ../../output/attr_models/attr_model_ub05 --percent 0.5 --train --feature ub
#python Attributes_Classifier.py ../../processed_data/attr_data_train.json ../../output/attr_models/attr_model_ub03 --percent 0.3 --train --feature ub

rm ../../output/attr_models/dev_results.txt


echo attr_model_u08 >> ../../output/attr_models/dev_results.txt
python Attributes_Classifier.py ../../processed_data/attr_data_dev.json ../../output/attr_models/attr_model_u08 --test >> ../../output/attr_models/dev_results.txt
echo attr_model_u05 >> ../../output/attr_models/dev_results.txt
python Attributes_Classifier.py ../../processed_data/attr_data_dev.json ../../output/attr_models/attr_model_u05 --test >> ../../output/attr_models/dev_results.txt
echo attr_model_u03 >> ../../output/attr_models/dev_results.txt
python Attributes_Classifier.py ../../processed_data/attr_data_dev.json ../../output/attr_models/attr_model_u03 --test >> ../../output/attr_models/dev_results.txt

echo attr_model_ub08 >> ../../output/attr_models/dev_results.txt
python Attributes_Classifier.py ../../processed_data/attr_data_dev.json ../../output/attr_models/attr_model_ub08 --test >> ../../output/attr_models/dev_results.txt
echo attr_model_ub05 >> ../../output/attr_models/dev_results.txt
python Attributes_Classifier.py ../../processed_data/attr_data_dev.json ../../output/attr_models/attr_model_ub05 --test >> ../../output/attr_models/dev_results.txt
echo attr_model_ub03 >> ../../output/attr_models/dev_results.txt
python Attributes_Classifier.py ../../processed_data/attr_data_dev.json ../../output/attr_models/attr_model_ub03 --test >> ../../output/attr_models/dev_results.txt
