#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
nsvc boosting method
first use sub_segments file to train a nsvc
then use the nsvc to find alignment in the dataset 
construct new training data sets
'''
import os
from New_Slot_value_classifier import *



def nsvc_boosting(model_dir, sub_segments, dataset, ontology_file, feature_list, tokenize_mode, use_stemmer, old_model_dir):
	# get svc model (load or train)
	svc = slot_value_classifier()
	if old_model_dir:
		if os.path.exists(old_model_dir):
			svc.LoadModel(old_model_dir)
			if not svc.is_set:
				raise Exception('Can not load model from :%s' %(old_model_dir))
	else:
		svc.TrainFromSubSegments(ontology_file, feature_list, sub_segments, model_dir, tokenize_mode, use_stemmer)

	# process dataset
	for call in dataset:
		for (log_utter, label_utter) in call:
			if 'frame_label' in label_utter:
				svc.appLogger.info('%d:%d\n'%(call.log['session_id'], log_utter['utter_index']))
				svc.appLogger.info('transcript: %s' %(log_utter['transcript']))
				frame_label = label_utter['frame_label']
				frame_tuples = svc.tuple_extractor.extract_tuple(frame_label)
				result, result_prob = svc.PredictUtter(log_utter, svc.feature.feature_list)
				tuple_results = []
				for key in frame_tuples:
					label = result[key]
					prob = result_prob[key][1]
					tuple_results.append((key, label, prob))
				for key, label, prob in tuple_results:
					svc.appLogger.info('%s, %d, %.3f' %(key, label, prob))






def main(argv):
	
	# 读取配置文件
	InitConfig()
	config = GetConfig()
	config.read([os.path.join(os.path.dirname(__file__),'../config/msiip_simple.cfg')])

	# 设置logging
	log_level_key = config.get('logging','level')
	run_code_name = os.path.basename(sys.argv[0])[0:-3]
	logging.basicConfig(filename = os.path.join(os.path.dirname(__file__), '../../output/logs', '%s_%s.log' %(run_code_name,time.strftime('%Y-%m-%d',time.localtime(time.time())))), \
    					level = GetLogLevel(log_level_key), 
    					format = '%(asctime)s %(levelname)8s %(lineno)4d %(module)s:%(name)s.%(funcName)s: %(message)s')
	

	parser = argparse.ArgumentParser(description='STC like slot value classifier.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--subseg', dest='subseg', action='store', help='The sub_segments file to analyze')
	parser.add_argument('model_dir',metavar='PATH', help='The output model dir')
	parser.add_argument('--old_model_dir',dest='old_model_dir',action='store', help='old model dir.')
	parser.add_argument('--ontology',dest='ontology',action='store', help='Ontology file.')
	parser.add_argument('--feature',dest='feature',action='store', help='feature to use. Example: TubB')
	parser.add_argument('--mode',dest='mode',action='store', help='tokenizer mode')
	parser.add_argument('--UseST',dest='UseST',action='store_true', help='use stemmer or not.')
	
	args = parser.parse_args()

	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)

	input = codecs.open(args.subseg, 'r', 'utf-8')
	sub_segments = json.load(input)
	input.close()

	feature_list = GetFeatureList(args.feature)

	nsvc_boosting(args.model_dir, sub_segments, dataset, args.ontology, feature_list, args.mode, args.UseST, args.old_model_dir)

if __name__ =="__main__":
	main(sys.argv)

