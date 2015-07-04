# TrainAndRunNSVC.sh u 5
set -u
set -e
./TrainNSVCModel.sh $1 $2
./msiip_nsvc_tracker.sh $1 $2
