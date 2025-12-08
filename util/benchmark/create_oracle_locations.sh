python -m util.benchmark.gen_oracle_locations \
    --repo_base_dir "/data/user_data/adityabs/playground/repo_base" \
    --output_dir "/data/user_data/adityabs" \
    --dataset "SWE-Gym/SWE-Gym" \
    --split "train" \
    --max_edit_file_num 1000 \
    --num_processes 16
