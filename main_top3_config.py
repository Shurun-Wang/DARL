import yaml
import os
import argparse


# ------------------------------ Args ------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='adhd', type=str, choices=['adhd', 'gad', 'idd', 'mdd', 'sch'])
    parser.add_argument('--model_name', default='OhCNN', type=str,
                        choices=['DeprNet', 'MBSzEEGNet', 'OhCNN', 'SzHNN', 'STGEFormer'])
    parser.add_argument('--algorithm', default='darl', type=str, help='tpe, rs, ga, rl, ppo, ahpo')
    return parser.parse_args()


def extract_top_configs(args, cur_path, input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    sorted_data = sorted(data, key=lambda x: x.get('val_acc', 0.0), reverse=True)
    top_3_records = sorted_data[:3]

    for i, record in enumerate(top_3_records):
        config = record.get('hp', {})
        path = cur_path + "/scripts/"+ args.model_name + "/"+ f"{args.algorithm}_{args.dataset}_{i+1}.yaml"
        with open(path, 'w', encoding='utf-8') as out_f:
            yaml.dump(config, out_f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    args = get_args()
    cur_path = os.path.dirname(os.path.abspath(__file__))
    input_yaml = cur_path + "/scripts/"+args.model_name+"/" + f"{args.algorithm}_{args.dataset}_history.yaml"
    extract_top_configs(args, cur_path, input_yaml)