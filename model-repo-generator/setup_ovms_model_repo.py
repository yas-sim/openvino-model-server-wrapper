import os
import sys
import glob
import json
import shutil
import argparse

def main(args):
    ir_model_dir = args.model_dir
    if not os.path.isdir(ir_model_dir):
        print('Source directory doesn\'t exist. \'{}\''.format(ir_model_dir))
        return -1

    # Search for IR models in the source directory
    xml_list = glob.glob(ir_model_dir+'/**/*.xml', recursive=True)

    # Check if corresponding '.bin' is existing.
    ir_models = []
    for xml in xml_list:
        path, ext = os.path.splitext(xml)
        if os.path.isfile(path+'.bin'):
            ir_models.append(xml)
        else:
            if args.verbose == True:
                print('\'{}\' is rejected because no corresponding .bin file exists.'.format(xml))
    if args.verbose: print()

    if len(ir_models)==0:
        print('No IR model found in \'{}\'.'.format(ir_model_dir))
        return -1

    if args.verbose == True:
        print(len(ir_models), 'IR models found.')
        for model in ir_models:
            print(model)
        print()

    # Generate model name and check name conflict
    # If a model name conflict is found, add parent direstory name as the postfix.
    # e.g. 'model' -> 'model-fp32'
    model_list = []
    for ir_model in ir_models:
        path, filename = os.path.split(ir_model)
        base, ext = os.path.splitext(filename)
        model_name = base
        for model in model_list:    # check if the same base name already exists
            if model['base'] == base:
                if model['model_name'] == base:   # check if the model name is the same as model_name to avoid multiple postfix addition
                    splitted_path = model['path'].split(os.sep)
                    model['model_name'] = (model['model_name'] + '-' + splitted_path[-1]).lower()  # add postfix
                splitted_path = path.split(os.sep)
                model_name = (base + '-' + splitted_path[-1]).lower()  # add postfix
        model_list.append({'path':path, 'filename':filename, 'base':base, 'ext':ext, 'model_name':model_name.lower()})

    ovms_model_repo_dir = args.output_dir

    # Copy IR models to repository. Construct config data.
    config = { 'model_config_list' : [] }
    for model in model_list:
        src_xml_name = model['base']+'.xml'
        src_bin_name = model['base']+'.bin'
        dst_xml_name = model['model_name']+'.xml'
        dst_bin_name = model['model_name']+'.bin'
        model_path = os.path.join(ovms_model_repo_dir, 'models', model['model_name'], '1')
        # Create directry and copy an IR model
        if args.dryrun == False:
            os.makedirs(model_path, exist_ok=True)
        if args.verbose:
            print('MKDIR {}'.format(model_path))
        if args.dryrun == False:
            shutil.copy(os.path.join(model['path'], src_xml_name), os.path.join(model_path, dst_xml_name))
        if args.verbose:
            print('CP {} {}'.format(os.path.join(model['path'], src_xml_name), os.path.join(model_path, dst_xml_name)))
        if args.dryrun == False:
            shutil.copy(os.path.join(model['path'], src_bin_name), os.path.join(model_path, dst_bin_name))
        if args.verbose:
            print('CP {} {}'.format(os.path.join(model['path'], src_bin_name), os.path.join(model_path, dst_bin_name)))

        ovms_model_path = os.path.join('/opt/models/', model['model_name'])    # model path from OVMS container perspective
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
    if args.verbose: print()

    # Write out the config data as a JSON file.
    cfg = json.loads(str(config).replace('\'', '"'))
    config_file_name = os.path.join(ovms_model_repo_dir, 'models', 'config.json')
    if args.verbose:
        json.dump(cfg, sys.stdout, indent=4)
        print()
    if args.dryrun == False:
        with open(config_file_name, 'wt') as f:
            json.dump(cfg, f, indent=4)
            print('OVMS model repository has been created in \'{}\'. \'{}\' configuration file is created.'.format(ovms_model_repo_dir, config_file_name))
    else:
        print('Dryrun completed.')

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVINO model server repository setup automation tool')
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='Source directory name that contains OpenVINO IR models.')
    parser.add_argument('-o', '--output_dir', type=str, default='./model_repository', help='OpenVINO-model-server model repository directory name to generate.')
    parser.add_argument('--dryrun', action='store_true', default=False, help='Run without any write')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose flag')
    args = parser.parse_args()
    sys.exit(main(args))
