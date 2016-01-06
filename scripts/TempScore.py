import sys,os,argparse,json,math
from ontology_reader import OntologyReader

SCHEDULES = [1,2]

def main(argv):
    install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_dirname = os.path.join(install_path,'lib')

    sys.path.append(utils_dirname)
    from dataset_walker import dataset_walker
    list_dir = os.path.join(install_path,'config')

    parser = argparse.ArgumentParser(description='Evaluate output from a belief tracker.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile',dest='trackfile',action='store',metavar='JSON_FILE',help='File containing tracker JSON output')
    parser.add_argument('--scorefile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,help='File to write with JSON scoring data')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')

    args = parser.parse_args()

    sessions = dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)

    if args.trackfile:
        tracker_output = json.load(open(args.trackfile))
    else:
        tracker_output = None

    ontology = OntologyReader(args.ontology)
    tagsets = ontology.get_tagsets()


    stats = {}
    stats[('all','all','all')] = [0,0,0]
    for topic, slots in tagsets.items():
        stats[(topic, 'all', 'all')] = [0,0,0]
        for slot in slots:
            stats[(topic, slot, 'all')] = [0,0,0]


    utter_counter = 0.0

    if tracker_output:
        for session, track_session in zip(sessions, tracker_output["sessions"]):
            session_id = session.log['session_id']

            prev_ref_frame = None
            prev_track_frame = None
            prev_topic = None

            for (log_utter, label_utter), track_utter in zip(session, track_session["utterances"]):
                utter_counter += 1.0

                if log_utter['segment_info']['target_bio'] == 'B':
                    # Beginning of a new segment
                    ref_frame = label_utter['frame_label']
                    track_frame = track_utter['frame_label']

                    if prev_ref_frame and prev_track_frame:
                        for slot, values in prev_ref_frame.items():
                            for value in values:
                                if (prev_topic,slot,value) not in stats:
                                    stats[(prev_topic,slot,value)] = [0,0,0]
                                stats[(prev_topic,slot,value)][1] += 1
                                stats[('all','all','all')][1] += 1
                                stats[(prev_topic,'all','all')][1] += 1
                                stats[(prev_topic,slot,'all')][1] += 1
                                if slot in prev_track_frame and value in prev_track_frame[slot]:
                                    stats[('all','all','all')][2] += 1
                                    stats[(prev_topic,'all','all')][2] += 1
                                    stats[(prev_topic,slot,'all')][2] += 1
                                    stats[(prev_topic,slot,value)][2] += 1

                        for slot, values in prev_track_frame.items():
                            for value in values:
                                if (prev_topic,slot,value) not in stats:
                                    stats[(prev_topic,slot,value)] = [0,0,0]
                                stats[('all','all','all')][0] += 1
                                stats[(prev_topic,'all','all')][0] += 1
                                stats[(prev_topic,slot,'all')][0] += 1
                                stats[(prev_topic,slot,value)][0] += 1


                    prev_ref_frame = ref_frame
                    prev_track_frame = track_frame
                    prev_topic = log_utter['segment_info']['topic']

                elif log_utter['segment_info']['target_bio'] == 'I':
                    ref_frame = label_utter['frame_label']
                    track_frame = track_utter['frame_label']

                    prev_ref_frame = ref_frame
                    prev_track_frame = track_frame
                    prev_topic = log_utter['segment_info']['topic']

                elif log_utter['segment_info']['target_bio'] == 'O':
                    ref_frame = None
                    track_frame = None

            if prev_ref_frame and prev_track_frame:
                for slot, values in prev_ref_frame.items():
                    for value in values:
                        if (prev_topic,slot,value) not in stats:
                            stats[(prev_topic,slot,value)] = [0,0,0]
                        stats[(prev_topic,slot,value)][1] += 1
                        stats[('all','all','all')][1] += 1
                        stats[(prev_topic,'all','all')][1] += 1
                        stats[(prev_topic,slot,'all')][1] += 1
                        if slot in prev_track_frame and value in prev_track_frame[slot]:
                            stats[('all','all','all')][2] += 1
                            stats[(prev_topic,'all','all')][2] += 1
                            stats[(prev_topic,slot,'all')][2] += 1
                            stats[(prev_topic,slot,value)][2] += 1

                for slot, values in prev_track_frame.items():
                    for value in values:
                        if (prev_topic,slot,value) not in stats:
                            stats[(prev_topic,slot,value)] = [0,0,0]
                        stats[('all','all','all')][0] += 1
                        stats[(prev_topic,'all','all')][0] += 1
                        stats[(prev_topic,slot,'all')][0] += 1
                        stats[(prev_topic,slot,value)][0] += 1
    else:
        for session in sessions:
            session_id = session.log['session_id']

            prev_ref_frame = None
            prev_topic = None

            for (log_utter, label_utter) in session:
                utter_counter += 1.0

                if log_utter['segment_info']['target_bio'] == 'B':
                    # Beginning of a new segment
                    ref_frame = label_utter['frame_label']

                    if prev_ref_frame:
                        for slot, values in prev_ref_frame.items():
                            for value in values:
                                if (prev_topic,slot,value) not in stats:
                                    stats[(prev_topic,slot,value)] = [0,0,0]

                                stats[('all','all','all')][1] += 1
                                stats[(prev_topic,'all','all')][1] += 1
                                stats[(prev_topic,slot,'all')][1] += 1
                                stats[(prev_topic,slot,value)][1] += 1

                    prev_ref_frame = ref_frame
                    prev_topic = log_utter['segment_info']['topic']

                elif log_utter['segment_info']['target_bio'] == 'I':
                    ref_frame = label_utter['frame_label']

                    prev_ref_frame = ref_frame
                    prev_topic = log_utter['segment_info']['topic']

                elif log_utter['segment_info']['target_bio'] == 'O':
                    ref_frame = None

            if prev_ref_frame:
                for slot, values in prev_ref_frame.items():
                    for value in values:
                        if (prev_topic,slot,value) not in stats:
                            stats[(prev_topic,slot,value)] = [0,0,0]

                        stats[('all','all','all')][1] += 1
                        stats[(prev_topic,'all','all')][1] += 1
                        stats[(prev_topic,slot,'all')][1] += 1
                        stats[(prev_topic,slot,value)][1] += 1


            

    csvfile = open(args.scorefile,'w')
    print >> csvfile,("topic, slot, value, tracker, ref, matched")

    for (topic,slot,value), values in stats.items():
        print >>csvfile,("%s, %s, %s, %d, %d, %d"%(topic, slot, value, values[0], values[1], values[2]))
    csvfile.close()

if (__name__ == '__main__'):
    main(sys.argv)
