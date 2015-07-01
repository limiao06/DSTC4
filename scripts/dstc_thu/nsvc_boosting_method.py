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



def nsvc_boosting(model_dir, sub_segments, dataset, ontology_file, feature_list, tokenize_mode, use_stemmer, old_model_dir=None, iteration = 1):
	# get svc model (load or train)
	svc = slot_value_classifier()
	if old_model_dir:
		if os.path.exists(old_model_dir):
			svc.LoadModel(old_model_dir)
			if not svc.is_set:
				raise Exception('Can not load model from :%s' %(old_model_dir))
	else:
		svc.TrainFromSubSegments(ontology_file, feature_list, sub_segments, old_model_dir, tokenize_mode, use_stemmer)

	# read old train data
	input = codecs.open(os.path.join(old_model_dir,'train_samples.json'))
	in_json = json.load(input)
	input.close()

	old_train_samples = in_json['train_samples']
	old_label_samples = in_json['label_samples']
	old_train_feature_samples = in_json['train_feature_samples']
	old_train_labels = in_json['train_labels']

	

	# process dataset
	for it in range(iteration):
		print 'iteration: %d' %(it)
		it_train_samples = []
		it_label_samples = []

		sub_segments_vec = []
		for call in dataset:
			for (log_utter, label_utter) in call:
				sys.stderr.write('%d:%d\n'%(call.log['session_id'], log_utter['utter_index']))
				if 'frame_label' in label_utter:
					if log_utter['segment_info']['target_bio'] == 'B':
						if sub_segments_vec:
							slot_value_dict = process_sub_segments_vec(sub_segments_vec, svc)
							sub_segments_vec = []
							# add train samples
							raw_input('press any thing to continue..')
					svc.appLogger.info('%d:%d'%(call.log['session_id'], log_utter['utter_index']))
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

					# add to sub_segments_vec
					sub_segments_vec.append((log_utter, label_utter, result_prob))

				else:
					if sub_segments_vec:
						slot_value_dict = process_sub_segments_vec(sub_segments_vec, svc)
						sub_segments_vec = []
						raw_input('press any thing to continue..')
						# add train samples

	
def process_sub_segments_vec(sub_segments_vec, svc, prob_threshold = 0.8):
	'''
	input a sub_segments_vec
	return a dict indicate which sent id correspond to a slot-value pair
	'''
	if not sub_segments_vec:
		return {}
	frame_label = sub_segments_vec[0][1]['frame_label']
	slot_value_dict = {}
	for slot in frame_label:
		for value in frame_label[slot]:
			key = str({slot:value})
			slot_value_dict[key] = []
	for key in slot_value_dict:
		score_vec = [0] * len(sub_segments_vec)
		for i, (log_utter, label_utter, result_prob) in enumerate(sub_segments_vec):
			if "ACK" in label_utter['speech_act'][0]['attributes']:
				pass
			else:
				t_frame_label = eval(key)
				tuples = svc.tuple_extractor.extract_tuple(t_frame_label)
				score = 0.0
				count = 0
				if svc.tuple_extractor.enumerable(t_frame_label.keys()[0]):
					for t in tuples:
						if t in result_prob and not t.startswith('root'):
							score += result_prob[t][1]
							count += 1
				else:
					for t in tuples:
						if t in result_prob:
							score += result_prob[t][1]
							count += 1
				score_vec[i] = score/count
		svc.appLogger.debug('key: %s, score_vec: %s' %(key, score_vec.__str__()))
		
		max_score = 0.0
		max_id = 0
		add_num = 0
		for i, score in enumerate(score_vec):
			if score > max_score:
				max_score = score
				max_id = i
			if score > prob_threshold:
				slot_value_dict[key].append(i)
				add_num += 1
		if add_num == 0:
			slot_value_dict[key].append(max_id)
		svc.appLogger.debug('key: %s, choose ids: %s' %(key, slot_value_dict[key].__str__()))
	return slot_value_dict



				







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
	parser.add_argument('--it',dest='iteration',action='store', type=int, help='iteration num.')
	
	args = parser.parse_args()

	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)

	input = codecs.open(args.subseg, 'r', 'utf-8')
	sub_segments = json.load(input)
	input.close()

	feature_list = GetFeatureList(args.feature)

	nsvc_boosting(args.model_dir, sub_segments, dataset, args.ontology, feature_list, args.mode, args.UseST, args.old_model_dir, args.iteration)

if __name__ =="__main__":
	main(sys.argv)

