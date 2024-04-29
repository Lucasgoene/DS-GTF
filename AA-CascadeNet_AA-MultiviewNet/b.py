from data_utils import download_batch_subjects, get_raw_data
import os

personal_access_key_id = '' 
secret_access_key = ''
hcp_path = os.getcwd()
#state_types = ["rest", "task_working_memory", "task_story_math", "task_motor"]
get_raw_data(subject='105923', hcp_path=hcp_path, type_state="task_story_math")

#download_batch_subjects(['140117'], personal_access_key_id, secret_access_key, hcp_path)