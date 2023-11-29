#! /usr/bin/bash
set -e

echo "$PWD"
#installing raspberry hard dependencies
sudo apt-get install build-essential libcap-dev



#installing picamera2
#sudo apt install -y python3-libcamera python3-kms++
sudo apt install -y python3-picamera2
#
#
#sudo apt install -y python3-pyqt5 python3-prctl libatlas-base-dev ffmpeg python3-pip


echo "$PWD"
#activating raspvenv
source "./raspvenv/bin/activate"


#pip3 install numpy --upgrade
#pip3 install picamera2



mapfile -t deps < "rasp-requirements.txt"
echo -e "pkgs in rasp-requirements.txt \n -${deps[@]}"



full_deps="${deps[@]:0:${#deps[@]}-3}";
echo -e "Full dependecy deps \n${full_deps[@]}"

printf  '%s\n' ${full_deps[@]} > temp_full_requirements.txt

no_deps="${deps[ -1]}";
echo -e "install with no dependencies \n${no_deps[@]}"

pip install -r temp_full_requirements.txt

rm temp_full_requirements.txt

pip install --no-deps ${no_deps[@]}

