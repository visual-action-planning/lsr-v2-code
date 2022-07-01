from __future__ import print_function
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import cv2
import pickle
import random
from os import path
from lsrv2_examples_utils import get_example_vap, perform_lsrv2_building, perform_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example' , type=str, default='rigid_normal', help='Type of task: rigid_normal, rigid_hard or hybrid')
    parser.add_argument('--from_scratch' , type=bool, default=False, help='produce from scratch')
    args = parser.parse_args()

    example_type=args.example
    from_scratch=args.from_scratch

    if (not os.path.isdir("examples")):
        os.makedirs("examples")

    if example_type=='rigid_normal':
        print("******producing normal box stacking example******")
        # models
        train_dataset_name="box_stacking_normal_task_2500"
        holdout_dataset_name="box_stacking_normal_task_holdout"
        mapping_module="box_stacking_normal_task_MM"
        action_prediction_module="box_stacking_normal_task_APM"
        apm_seed = 1977
        
        #set hyperparams:
        distance_type=1
        c_max=1
        rng=random.randint(0,1337)
        
    elif example_type=='rigid_hard':
        print("******producing hard box stacking example******")
        # models:
        train_dataset_name="box_stacking_hard_task_2500"
        holdout_dataset_name="box_stacking_hard_task_holdout"
        mapping_module="box_stacking_hard_task_MM"
        action_prediction_module="box_stacking_hard_task_APM"
        apm_seed = 1977
        
        # set hyperparams:
        distance_type=1
        c_max=20
        rng=random.randint(0,1337)
        
    elif example_type=='hybrid':
        print("******producing rope box manipulation example******")
        #models:
        train_dataset_name="rope_box_task_2500"
        holdout_dataset_name="rope_box_task_holdout"
        mapping_module="rope_box_task_MM"
        action_prediction_module="rope_box_task_APM"
        apm_seed = 1977
        
        #set hyperparams:
        distance_type=1
        c_max=20
        rng=random.randint(0,1337)
        
    else:
        assert False, "Not a valid example type! Use --example=rigid_normal, rigid_hard or hybrid"


    # map training dataset with mapping module
    mm_output_file="./examples/mm_" + mapping_module + "_data_" + train_dataset_name + ".pkl"
    if from_scratch or not path.exists(mm_output_file):
        latent_map= perform_mapping(train_dataset_name,mapping_module)        
        # save
        with open(mm_output_file, 'wb') as f:
            pickle.dump(latent_map, f)
        print("--- mapping done ---")
    else:
        print(" found: " + mm_output_file + " already exist, using it.")
        file = open(mm_output_file,'rb')
        latent_map = pickle.load(file)
        file.close()

    # build lsrv2
    lsr_output_name="./examples/lsrv2_" +  mapping_module + "_data_" + train_dataset_name + ".pkl"
    if from_scratch or not path.exists(lsr_output_name):
        lsrv2=perform_lsrv2_building(latent_map,distance_type,c_max)
        # save
        with open(lsr_output_name, 'wb') as f:
            pickle.dump(lsrv2, f)
        print("--- lsrv2 building done ---")
        

    else:
        print(" found: " + lsr_output_name + " already exists, using it.")
        # load
        file = open(lsr_output_name,'rb')
        lsrv2 = pickle.load(file)
        file.close()




    example_visual_action_plan=get_example_vap(example_type,lsrv2,mapping_module,action_prediction_module,
                                               apm_seed,holdout_dataset_name,rng)

     
    # save
    cv2.imwrite(example_type + ".png",example_visual_action_plan)
    print("**** produced " + example_type + ".png")


if __name__== "__main__":
  main()