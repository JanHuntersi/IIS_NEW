# this function will make a dir if not exists
import os


def make_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path 
