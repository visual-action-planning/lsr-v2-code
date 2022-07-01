from dataloader import preprocess_triplet_data_seed

def main():

    dataset_names = ["box_stacking_normal_task_2500",
                     "box_stacking_hard_task_2500",
                     "rope_box_task_2500"]
    seeds = [1122]
    for dataset_name in dataset_names:
    	for seed in seeds:
    		preprocess_triplet_data_seed(dataset_name + '.pkl', seed)  

if __name__== "__main__":
  main()
