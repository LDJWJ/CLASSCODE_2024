%%bash
export PROJECT="apt-diode-156508"
export REGION="us-east1"
gcloud config set project $PROJECT
gcloud config set compute/region $REGION

## chmod 755 [쉘명]
## ./[쉘명]
## 결과
#./step03.sh: line 3: fg: no job control
#Updated property [core/project].
#Updated property [compute/region].
