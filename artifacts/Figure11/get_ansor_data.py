import os.path as osp
import subprocess

sub_dirs = [
    ["NAFNet/64"],
    ["vit/64"],
    ["swin/64"],
]

def get_sub_dirs(prefix):
    results = []
    model_strings = []
    for vals in sub_dirs:
        model_strings.append(f"Model: {vals[0]}")
        results.append(osp.join(prefix, *map(str, vals)))
    return results, model_strings

if __name__ == "__main__":
    prefix = "/sharepoint/e2e"
    for sub_dir, model_string in zip(*get_sub_dirs(prefix)):
        print("Running", model_string)
        base_comamnd = "python run_ansor_profiled.py --prefix " + sub_dir
        command1 = "nvprof --profile-from-start off --log-file profile --csv " + base_comamnd
        subprocess.run(command1.split(), check=True)
        command2 = "nvprof --profile-from-start off --log-file metrics --csv --metrics gld_throughput,gst_throughput,flop_count_sp " + base_comamnd
        subprocess.run(command2.split(), check=True)
        command3 = "python process_metrics.py"
        subprocess.run(command3.split(), check=True)
        command4 = "python run_ansor.py --prefix " + sub_dir
        subprocess.run(command4.split(), check=True)