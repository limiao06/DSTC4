#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
nsvc boosting method
first use sub_segments file to train a nsvc
then use the nsvc to find alignment in the dataset 
construct new training data sets


new things:
1. add output alignment file function
2. Now label_sample has positive case and NpNn(Neither positive Nor negative) case, NpNn case will not used in SVM training data
    only label_sample generated from boosting method may be a dict instance, any other is a list instance

'''
import os
from New_Slot_value_classifier import *
from Utils import *
from random import random



def nsvc_boosting(model_dir, sub_segments, dataset, ontology_file, feature_list, \
				 tokenizer_mode, use_stemmer, remove_stopwords, \
				 old_model_dir=None, iteration = 1, \
				 high_thres=0.8, low_thres=0.2, alpha=0.2):
	# get svc model (load or train)
	svc = slot_value_classifier()
	if old_model_dir:
		if os.path.exists(old_model_dir):
			svc.LoadModel(old_model_dir)
			if not svc.is_set:
				raise Exception('Can not load model from :%s' %(old_model_dir))
	else:
		svc.TrainFromSubSegments(ontology_file, feature_list, sub_segments, old_model_dir, \
								 tokenizer_mode, use_stemmer, remove_stopwords)

	# read old train data
	input = codecs.open(os.path.join(old_model_dir,'train_samples.json'))
	in_json = json.load(input)
	input.close()

	old_train_samples = in_json['train_samples']
	old_label_samples = in_json['label_samples']
	old_train_feature_samples = in_json['train_feature_samples']
	old_train_labels = in_json['train_labels']

	

	# boosting

	if os.path.exists(model_dir):
		shutil.rmtree(model_dir,True)
	os.mkdir(model_dir)

	for it in range(iteration):
		print 'iteration: %d' %(it)
		it_train_samples = []
		it_label_samples = []

		# generate new train samples
		sub_segments_vec = []

		alignment_dict = {}
		'''
		alignment_dict stores the alignment information,
		key of this dict is a tuple (session_id, utter_index)
		value of this dict is a frame_label
		'''
		for call in dataset:
			for (log_utter, label_utter) in call:
				sys.stderr.write('%d:%d\n'%(call.log['session_id'], log_utter['utter_index']))
				if 'frame_label' in label_utter:
					if log_utter['segment_info']['target_bio'] == 'B':
						if sub_segments_vec:
							slot_value_dict = choose_from_sub_segments_vec(sub_segments_vec, svc, high_thres, low_thres, alpha)
							# generate alignment data
							session_id = call.log['session_id']
							for key in slot_value_dict:
								for i in slot_value_dict[key]["positive"]:
									utter_index = sub_segments_vec[i][0]['utter_index']
									align_key = (session_id, utter_index)
									if align_key not in alignment_dict:
										alignment_dict[align_key] = {}
									t_frame_label = eval(key)
									for slot in t_frame_label:
										if slot not in alignment_dict[align_key]:
											alignment_dict[align_key][slot] = []
										for value in t_frame_label[slot]:
											if value not in alignment_dict[align_key][slot]:
												alignment_dict[align_key][slot].append(value)


							# generate new training samples
							new_label_samples, new_train_samples = generate_sample_from_sub_segments_vec(sub_segments_vec, slot_value_dict, svc)
							for l_sample, t_sample in zip(new_label_samples, new_train_samples):
								svc.appLogger.debug('new sample: (%s, %s)' %(l_sample.__str__(), t_sample.__str__()))
							it_train_samples.extend(new_train_samples)
							it_label_samples.extend(new_label_samples)
							sub_segments_vec = []
							#raw_input('press any thing to continue.')
							
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

					#for key, label, prob in tuple_results:
					#	svc.appLogger.info('%s, %d, %.3f' %(key, label, prob))

					# add to sub_segments_vec
					sub_segments_vec.append((log_utter, label_utter, result_prob))

				else:
					if sub_segments_vec:
						slot_value_dict = choose_from_sub_segments_vec(sub_segments_vec, svc, high_thres, low_thres, alpha)
						# generate alignment data
						session_id = call.log['session_id']
						for key in slot_value_dict:
							for i in slot_value_dict[key]["positive"]:
								utter_index = sub_segments_vec[i][0]['utter_index']
								align_key = (session_id, utter_index)
								if align_key not in alignment_dict:
									alignment_dict[align_key] = {}
								t_frame_label = eval(key)
								for slot in t_frame_label:
									if slot not in alignment_dict[align_key]:
										alignment_dict[align_key][slot] = []
									for value in t_frame_label[slot]:
										if value not in alignment_dict[align_key][slot]:
											alignment_dict[align_key][slot].append(value)


						# generate new training samples
						new_label_samples, new_train_samples = generate_sample_from_sub_segments_vec(sub_segments_vec, slot_value_dict, svc)
						for l_sample, t_sample in zip(new_label_samples, new_train_samples):
							svc.appLogger.debug('new sample: (%s, %s)' %(l_sample.__str__(), t_sample.__str__()))
						it_train_samples.extend(new_train_samples)
						it_label_samples.extend(new_label_samples)
						sub_segments_vec = []
						#raw_input('press any thing to continue.')

		# train new models
		feature_list = svc.feature.feature_list
		tokenizer_mode = svc.feature.tokenizer_mode
		use_stemmer = svc.feature.use_stemmer
		remove_stopwords = svc.feature.remove_stopwords
		'''
		# train model only based on new aligned data
		new_model_dir = os.path.join(model_dir, 'NA'+str(it))
		if os.path.exists(new_model_dir):
			shutil.rmtree(new_model_dir,True)
		os.mkdir(new_model_dir)
		svc._train_by_samples(new_model_dir, it_label_samples, it_train_samples, feature_list, tokenizer_mode, use_stemmer, remove_stopwords)
		'''
		# train model based on old data and new aligned data
		new_label_samples = old_label_samples + it_label_samples
		new_train_samples = old_train_samples + it_train_samples

		new_model_dir = os.path.join(model_dir, str(it))
		if os.path.exists(new_model_dir):
			shutil.rmtree(new_model_dir,True)
		os.mkdir(new_model_dir)
		svc._train_by_samples(new_model_dir, new_label_samples, new_train_samples, svc.feature.feature_list, svc.feature.tokenizer_mode, svc.feature.use_stemmer, svc.feature.remove_stopwords)
		del svc
		svc = slot_value_classifier()
		svc.LoadModel(new_model_dir)
		if not svc.is_set:
			raise Exception('Can not load model from :%s' %(new_model_dir))
		# save alignment dict
		out_alignment_file = os.path.join(new_model_dir, 'alignment_dict.json')
		out_alignment_dict = TupleKeyDict2StrKeyDict(alignment_dict)
		output = file(out_alignment_file, 'w')
		json.dump(out_alignment_dict, output, indent=4)
		output.close()

	print 'done!'

'''
# backup
def generate_sample_from_sub_segments_vec(sub_segments_vec, slot_value_dict, svc):
	train_samples = []
	label_samples = []
	for i, (log_utter, label_utter, result_prob) in enumerate(sub_segments_vec):
		sample = svc._extract_utter_tuple(log_utter, svc.feature.feature_list)
		label_sample = []
		for key in slot_value_dict:
			if i in slot_value_dict[key]:
				tuples = svc.tuple_extractor.extract_tuple(eval(key))
				label_sample.extend(tuples)
		train_samples.append(sample)
		label_samples.append(label_sample)
	return (label_samples, train_samples)
'''

def generate_sample_from_sub_segments_vec(sub_segments_vec, slot_value_dict, svc):
	train_samples = []
	label_samples = []
	for i, (log_utter, label_utter, result_prob) in enumerate(sub_segments_vec):
		sample = svc._extract_utter_tuple(log_utter, svc.feature.feature_list)
		label_sample = {"positive":[], "NpNn": []}
		for key in slot_value_dict:
			if i in slot_value_dict[key]["positive"]:
				tuples = svc.tuple_extractor.extract_tuple(eval(key))
				label_sample["positive"].extend(tuples)
			if i in slot_value_dict[key]["NpNn"]:
				tuples = svc.tuple_extractor.extract_tuple(eval(key))
				label_sample["NpNn"].extend(tuples)
		train_samples.append(sample)
		label_samples.append(label_sample)
	return (label_samples, train_samples)
	
def choose_from_sub_segments_vec(sub_segments_vec, svc, high_thres, low_thres, alpha):
	'''
	input a sub_segments_vec
	return a dict indicate which sent id correspond to a slot-value pair

	alpha is a weight factor, for a slot value pair (s,v) and its correspond tuples [root:s, s:v]
		alpha indicate which is important to calc score root:s or s:v
	'''
	if not sub_segments_vec:
		return {}
	frame_label = sub_segments_vec[0][1]['frame_label']
	slot_value_dict = {}
	for slot in frame_label:
		for value in frame_label[slot]:
			key = str({slot:[value]})
			slot_value_dict[key] = {"positive":[], "NpNn": []}
	for key in slot_value_dict:
		svc.appLogger.debug('start processing key: %s' %(key))
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
						if t in result_prob:
							if t.startswith('root'):
								score += result_prob[t][1] * alpha
								count += alpha
							else:
								score += result_prob[t][1] * (1- alpha)
								count += 1- alpha
				else:
					for t in tuples:
						if t in result_prob:
							score += result_prob[t][1]
							count += 1
				score_vec[i] = score/count
		svc.appLogger.debug('key: %s, score_vec: %s' %(key, score_vec.__str__()))
		'''
		max_score = 0.0
		max_id = 0
		add_num = 0
		for i, score in enumerate(score_vec):
			if score > max_score:
				max_score = score
				max_id = i
			if score > high_thres:
				slot_value_dict[key]["positive"].append(i)
				add_num += 1
		if add_num == 0:	# if there is no case whose score > high_thres, then add the case with max score to positive
			slot_value_dict[key]["positive"].append(max_id)

		# add NpNn case
		for i, score in enumerate(score_vec):
			if score > low_thres and score < high_thres and i != max_id:
				slot_value_dict[key]["NpNn"].append(i)
		'''

		# change choose samples by random
		for i, score in enumerate(score_vec):
			if score > low_thres:
				rd = random()
				if rd <= score:
					slot_value_dict[key]["positive"].append(i)
				else:
					slot_value_dict[key]["NpNn"].append(i)

		svc.appLogger.debug('key: %s, choose positive ids: %s' %(key, slot_value_dict[key]["positive"].__str__()))
		svc.appLogger.debug('key: %s, choose Neither positive nor negative ids: %s' %(key, slot_value_dict[key]["NpNn"].__str__()))
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
	parser.add_argument('--RemoveSW',dest='RemoveSW',action='store_true', help='Remove stop words or not.')	
	parser.add_argument('--it',dest='iteration',action='store', type=int, help='iteration num.')
	parser.add_argument('--ht',dest='high_thres',action='store', type=float, default=0.8, help='High threshold parameter of boosting.')
	parser.add_argument('--lt',dest='low_thres',action='store', type=float, default=0.2, help='High threshold parameter of boosting.')
	
	args = parser.parse_args()

	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)

	input = codecs.open(args.subseg, 'r', 'utf-8')
	sub_segments = json.load(input)
	input.close()

	feature_list = GetFeatureList(args.feature)

	nsvc_boosting(args.model_dir, sub_segments, dataset, args.ontology, feature_list,\
				 args.mode, args.UseST, args.RemoveSW,\
				 args.old_model_dir, args.iteration,\
				 args.high_thres, args.low_thres)

if __name__ =="__main__":
	main(sys.argv)

