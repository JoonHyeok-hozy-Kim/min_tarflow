import os
import glob


def get_best_checkpoint(weight_dir):
    assert os.path.exists(weight_dir), f"Invalid path : {weight_dir}"
    final_format = weight_dir + "/model_final.pth"
    model_final_file = glob.glob(final_format)
    assert not model_final_file, f"Final already exists at {weight_dir}"
    
    best_format = weight_dir + "/model_best.pth"
    model_best_file = glob.glob(best_format)
    assert model_best_file, f"Best model not found at {weight_dir}"
    
    return model_best_file[0]


def get_wandb_run_info(run_url):
    # Get Info
    elements = run_url.split("//")[1].split("/")
    user_name = elements[1]
    project_name = elements[2]
    run_id = elements[4].split("?")[0]
    
    return user_name, project_name, run_id    