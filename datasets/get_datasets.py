from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="box_stacking_normal_task_2500.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="box_stacking_normal_task_holdout.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="box_stacking_hard_task_2500.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="box_stacking_hard_task_holdout.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="rope_box_task_2500.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
hf_hub_download(repo_id="LSR-FG/lsr-datasets", filename="rope_box_task_holdout.pkl", repo_type="dataset", local_dir=".",local_dir_use_symlinks=False)
print("donzo")