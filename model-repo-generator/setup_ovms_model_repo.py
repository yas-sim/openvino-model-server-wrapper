import os
import sys
import glob
import json
import shutil
import argparse

def main(args):
    ir_model_dir = args.model_dir
    ovms_model_repo_dir = args.output_dir

    config = { 'model_config_list' : [] }

    ir_models = glob.glob(ir_model_dir+'/**/*.xml', recursive=True)

    # Generate model name and check name duplication
    # If a duplicate is found, add parent direstory name as the postfix. (model -> model-fp32)
    model_list = []
    for ir_model in ir_models:
        path, filename = os.path.split(ir_model)
        base, ext = os.path.splitext(filename)
        model_name = base
        for model in model_list:
            if model['base'] == base:
                if model['model_name'] == base:
                    splitted_path = model['path'].split(os.sep)
                    model['model_name'] = (model['model_name'] + '-' + splitted_path[-1]).lower()
                splitted_path = path.split(os.sep)
                model_name = (base + '-' + splitted_path[-1]).lower()
        model_list.append({'path':path, 'filename':filename, 'base':base, 'ext':ext, 'model_name':model_name.lower()})

    # Copy IR models to repository. Construct config data.
    for model in model_list:
        src_xml_name = model['base']+'.xml'
        src_bin_name = model['base']+'.bin'
        dst_xml_name = model['model_name']+'.xml'
        dst_bin_name = model['model_name']+'.bin'
        model_path = os.path.join(ovms_model_repo_dir, 'models', model['model_name'], '1')
        os.makedirs(model_path, exist_ok=True)
        shutil.copy(os.path.join(model['path'], src_xml_name), os.path.join(model_path, dst_xml_name))
        shutil.copy(os.path.join(model['path'], src_bin_name), os.path.join(model_path, dst_bin_name))
        ovms_model_path = os.path.join('/opt/models/', model['model_name'])
        cfg_str = { 
            'config': { 
                'name'          : model['model_name'],
                'base_path'     : ovms_model_path,
                "batch_size"    : "auto",
                "target_device" : "CPU",
                "plugin_config" : {},
                "nireq"         : 0,
                "model_version_policy" : {"all":{}},
            }
        }
        config['model_config_list'].append(cfg_str)

    # Write out the config data as a JSON file.
    cfg = json.loads(str(config).replace('\'', '"'))
    config_file_name = os.path.join(ovms_model_repo_dir, 'models', 'config.json')
    with open(config_file_name, 'wt') as f:
        json.dump(cfg, f, indent=4)

    print('OVMS model repository has been created in \'{}\'. \'{}\' configuration file is created.'.format(ovms_model_repo_dir, config_file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVINO model server repository setup automation tool')
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='Source directory name that contains OpenVINO IR models.')
    parser.add_argument('-o', '--output_dir', type=str, default='./model_repository', help='OpenVINO-model-server model repository directory name to generate.')
    args = parser.parse_args()
    sys.exit(main(args))
