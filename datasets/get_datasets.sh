#!/bin/bash 

# Simulation datasets

#download box stacking normal
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Zdg28MPb9hf-Pogc1tDYN-H5gwzY5hVg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Zdg28MPb9hf-Pogc1tDYN-H5gwzY5hVg" -O box_stacking_normal_task.tar.xz && rm -rf /tmp/cookies.txt
#unpack box stacking normal
tar -xf box_stacking_normal_task.tar.xz
#delete box stacking normal tar
rm box_stacking_normal_task.tar.xz


#download box stacking hard
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dWTgIuGE87-O5-A35VNZotdv25dHAjU8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dWTgIuGE87-O5-A35VNZotdv25dHAjU8" -O box_stacking_hard_task.tar.xz && rm -rf /tmp/cookies.txt
#unpack box stacking hard
tar -xf box_stacking_hard_task.tar.xz
#delete box stacking hard tar
rm box_stacking_hard_task.tar.xz


#download rope-box 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yXx526vOmrba9bCdvDhFg2IcVE0VIXXA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yXx526vOmrba9bCdvDhFg2IcVE0VIXXA" -O rope_box_task.tar.xz && rm -rf /tmp/cookies.txt
#unpack rope-box 
tar -xf rope_box_task.tar.xz
#delete rope-box  tar
rm rope_box_task.tar.xz


