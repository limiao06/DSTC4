# TrainAndRunNSVC.sh u 5
set -u
set -e
./TrainNSVCModelRS.sh $1 $2
./msiip_nsvc_tracker_RS.sh $1 $2
