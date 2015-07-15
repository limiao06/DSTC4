#! /usr/bin/python
# -*- coding: utf-8 -*- 

'''
nsvc dataset train or test tools
'''

from New_Slot_value_classifier import *


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
	parser.add_argument('--subseg', dest='subseg', action='store', required=True, help='The sub_segments file to analyze')
	parser.add_argument('model_dir',metavar='PATH', help='The output model dir')
	parser.add_argument('--train',dest='train',action='store_true', help='train or test.')
	parser.add_argument('--ontology',dest='ontology',action='store', help='Ontology file.')
	parser.add_argument('--feature',dest='feature',action='store', help='feature to use. Example: TubB')
	parser.add_argument('--mode',dest='mode',action='store', help='tokenizer mode')
	parser.add_argument('--UseST',dest='UseST',action='store_true', help='use stemmer or not.')
	parser.add_argument('--test',dest='test',action='store_true', help='train or test.')
	parser.add_argument('--RemoveSW',dest='RemoveSW',action='store_true', help='Remove stop words or not.')	
	args = parser.parse_args()

	svc = slot_value_classifier()

	input = codecs.open(args.subseg, 'r', 'utf-8')
	sub_segments = json.load(input)
	input.close()

	if args.test and args.train:
		sys.stderr.write('Error: train and test can not be both ture!')
	elif not (args.test or args.train):
		sys.stderr.write('Error: train and test can not be both false!')
	elif args.test:
		print 'Test!'
		svc.TestFromSubSegments(sub_segments, args.model_dir)
	else:
		print 'Train'
		feature_list = GetFeatureList(args.feature)
		svc.TrainFromSubSegments(args.ontology, feature_list, sub_segments, args.model_dir, args.mode, args.UseST, args.RemoveSW)

if __name__ =="__main__":
	main(sys.argv)

