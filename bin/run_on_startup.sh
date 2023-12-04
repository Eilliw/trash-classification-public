#!/bin/bash
DISPLAY=:0 xterm -hold -e /home/remote/Dev/Procam

#sudo killall pigpiod
sudo pigpiod
cd /home/project/GEEN1400-Final-Project
cd /home/project/trash-classification-public
source venv/bin/activate
python3 scripts/button.py
