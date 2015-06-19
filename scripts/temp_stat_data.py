import argparse, sys, ontology_reader, dataset_walker, time, json

from collections import Counter
from temp_extract_sub_segments import sub_segment_extractor

def extract_semantic_tags(utter):
	utter = utter.strip()
	sem_tags = []
	start_pos = 0
	while True:
		sem_tag = {}
		tag_start = utter.find('<', start_pos)
		if tag_start == -1:
			break

		if utter[tag_start+1] == '/':
			start_pos = tag_start + 2
			continue
		tag_end = utter.find('>', tag_start)
		sem_tag_str = utter[tag_start+1:tag_end]
		tokens = sem_tag_str.split(' ')
		sem_tag['name'] = tokens[0].strip()
		sem_tag['slots'] = {}
		for i in range(1, len(tokens)):
			equal_pos = tokens[i].find('=')
			slot = tokens[i][0:equal_pos].strip()
			value = tokens[i][equal_pos+2:-1].strip()
			sem_tag['slots'][slot] = value
		sem_tag['value'] = utter[tag_end+1:utter.find('<',tag_end+1)].strip()
		start_pos = utter.find('>', tag_end+1) + 1
		sem_tags.append(sem_tag)
	return sem_tags




def main(argv):
	parser = argparse.ArgumentParser(description='Stat information about the data.')
	parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
	parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
	parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
	parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')

	args = parser.parse_args()
	dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)
	tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()

	track_file = open(args.trackfile, "wb")
	track = {"sessions":[]}
	track["dataset"]  = args.dataset
	start_time = time.time()

	out_json = {}
	out_json['speech_act_cat_dic'] = {}
	out_json['speech_act_attr_dic'] = {}
	out_json['semantic_tags_dic'] = {}
	out_json['ontology_dic'] = {}

	for topic in tagsets:
		out_json['ontology_dic'][topic] = {'count':0, 'detail':{}}
		for slot, values in tagsets[topic].items():
			out_json['ontology_dic'][topic]['detail'][slot] = {'count':0, 'detail':{}}
			#for value in values:
			#	out_json['ontology_dic'][topic]['detail'][slot]['detail'][value] = 0	



	extractor = sub_segment_extractor()

	for call in dataset:
		extractor.reset()
		for (log_utter, label_utter) in call:
			sys.stderr.write('%d:%d\n'%(call.log['session_id'], log_utter['utter_index']))
			# speech act
			for sa in label_utter['speech_act']:
				act = sa['act'].strip()
				attrs = sa['attributes']
				if act not in out_json['speech_act_cat_dic']:
					out_json['speech_act_cat_dic'][act] = {'count':0, 'detail':{}}
				out_json['speech_act_cat_dic'][act]['count'] += 1
				for attr in attrs:
					attr = attr.strip()
					if attr not in out_json['speech_act_cat_dic'][act]['detail']:
						out_json['speech_act_cat_dic'][act]['detail'][attr] = 1
					else:
						out_json['speech_act_cat_dic'][act]['detail'][attr] += 1
					if attr not in out_json['speech_act_attr_dic']:
						out_json['speech_act_attr_dic'][attr] = 1
					else:
						out_json['speech_act_attr_dic'][attr] += 1

			# semantic 
			for semantic_tag in label_utter['semantic_tagged']:
				sem_tags = extract_semantic_tags(semantic_tag)
				for sem_tag in sem_tags:
					name = sem_tag['name']
					value = sem_tag['value']
					if name not in out_json['semantic_tags_dic']:
						out_json['semantic_tags_dic'][name] = {'count':0, 'detail':{}}
					out_json['semantic_tags_dic'][name]['count'] += 1
					if value not in out_json['semantic_tags_dic'][name]['detail']:
						out_json['semantic_tags_dic'][name]['detail'][value] = 1
					else:
						out_json['semantic_tags_dic'][name]['detail'][value] += 1

			topic = log_utter['segment_info']['topic']
			# frame label
			if log_utter['segment_info']['target_bio'] == 'B':
				if not extractor.is_empty:
					sub_segment = extractor.state
					topic = sub_segment['topic']
					out_json['ontology_dic'][topic]['count'] += 1

					for slot, value_list in sub_segment['frame_label'].items():
						out_json['ontology_dic'][topic]['detail'][slot]['count'] += 1
						for t_value in value_list:
							t_value = t_value.strip()
							if t_value not in out_json['ontology_dic'][topic]['detail'][slot]['detail']:
								out_json['ontology_dic'][topic]['detail'][slot]['detail'][t_value] = 1
							else:
								out_json['ontology_dic'][topic]['detail'][slot]['detail'][t_value] += 1
			extractor.addUtter(log_utter,label_utter)

		

	json.dump(out_json, track_file, indent=4)

	track_file.close()

if __name__ =="__main__":
	main(sys.argv)
