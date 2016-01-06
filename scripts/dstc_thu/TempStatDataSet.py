'''
State dataset, state the occurance of each slot-value pairs in a dataset
'''
import sys,os,argparse,json,math
import codecs

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from ontology_reader import OntologyReader

def main(argv):
    parser = argparse.ArgumentParser(description='State a dataset.')
    parser.add_argument('--subseg', dest='subseg', action='store', required=True, help='The sub_segments file to analyze')
    parser.add_argument('--outfile',dest='outfile',action='store',metavar='CSV_FILE',required=True,help='File to write with csv out data')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')

    args = parser.parse_args()

    input = codecs.open(args.subseg, 'r', 'utf-8')
    sub_segments = json.load(input)
    input.close()

    ontology = OntologyReader(args.ontology)

    tagsets = ontology.get_tagsets()

    stat = {}
    stat[('all', 'all')] = 0
    for topic in ontology.get_topics():
        for slot in ontology.get_slots(topic) + ['all']:
            stat[(topic, slot)] = 0

    for session in sub_segments['sessions']:
        for sub_seg in session['sub_segments']:
            topic = sub_seg['topic']
            frame_label = sub_seg['frame_label']
            if frame_label:
                stat[('all', 'all')] += 1
                stat[(topic, 'all')] += 1

                for slot,values in frame_label.items():
                    stat[(topic, slot)] += len(values)


    csvfile = open(args.outfile,'w')
    print >> csvfile,("topic,slot,count,value_number")
    print >> csvfile,("all,all,%d," %(stat[('all','all')]))
    for topic in ontology.get_topics():
        for slot in ontology.get_slots(topic):
            print >> csvfile,("%s,%s,%d,%d" %(topic, slot, stat[(topic,slot)], len(tagsets[topic][slot])))
        print >> csvfile,("%s,all,%d," %(topic, stat[(topic,'all')]))
    csvfile.close()

if (__name__ == '__main__'):
    main(sys.argv)
