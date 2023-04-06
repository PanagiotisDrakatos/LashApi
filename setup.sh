apt update -y
apt upgrade

apt install build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm

mkdir python_installation && cd python_installation

sudo apt-get update && sudo apt-get upgrade
sudo apt install python3
sudo apt install python3-pip
pip3 install virtualenv
virtualenv --version
virtualenv venv
source venv/bin/activate
python3 -m pip install pandas pyproj tensorflow tensorflow_recommenders Django djangorestframework requests geopy

