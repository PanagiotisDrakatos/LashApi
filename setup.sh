apt update -y
apt upgrade

apt install build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev -y

mkdir python_installation && cd python_installation

wget https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tgz
tar xzvf Python-3.7.2.tgz
rm -f Python-3.7.2.tgz

cd Python-3.7.2
./configure --enable-optimizations
make -j 4
make altinstall

cd ../..
rm -rf python_installation

apt --purge remove build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev -y
apt autoremove -y
apt clean

sudo apt-get install pipenv
pipenv install
pipenv shell
python3.7 -m pip install -U pip
python3.7 -m pip install pandas tensorflow Django djangorestframework requests
echo '$alias pip3="python3.7 -m pip"' >> ~/.bashrc