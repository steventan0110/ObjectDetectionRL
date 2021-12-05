import os
import shutil


def organize_resutls(dir):
    parent_folder_path = os.path.dirname(dir)
    folder_name_path = os.path.basename(dir)
    new_folder = os.path.join(parent_folder_path, f'{folder_name_path}_stats')
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    model_names = ['dqn', 'pretrained_dqn', 'dueling_dqn']
    exp_filenames = os.listdir(dir)
    for exp_filename in exp_filenames:
        for model_name in model_names:
            stats_folder = os.path.join(dir, exp_filename, model_name, 'stats')
            stats_json_names = os.listdir(stats_folder)
            if len(stats_json_names) == 1:
                stats_filename = stats_json_names[0]
                new_stats_folder = os.path.join(new_folder, exp_filename, model_name)
                os.makedirs(new_stats_folder)
                old_stats_file = os.path.join(stats_folder, stats_filename)
                new_stats_file = os.path.join(new_stats_folder, stats_filename)
                shutil.copyfile(old_stats_file, new_stats_file)
                print('COPIED:', old_stats_file)


if __name__ == '__main__':
    res_path = '/raid/home/slai16/ObjectDetectionRL/validation_lr_0_000001'
    organize_resutls(res_path)