import argparse
import subprocess
import time

import warnings
warnings.filterwarnings("ignore")

def run_python_script(args):
    # Start monitoring CPU and memory usage
    monitor_process = subprocess.Popen(["./compression/scripts/monitor_usage.sh"], stdout=subprocess.PIPE)

    if args.base:
        print("Model_PANN (1000):")
        subprocess.run(["python3", "-m", "compression.test_inference", "--base"])
    elif(args.qat):
        print("PANN_QAT (1000):")
        subprocess.run(["python3", "-m", "compression.test_inference", "--qat"])
    elif(args.qat2):
        print("PANN_QATv2 (1000):")
        subprocess.run(["python3", "-m", "compression.test_inference", "--qat2"])
    elif(args.sq):
        print("PANN_SQ (1000):")
        subprocess.run(["python3", "-m", "compression.test_inference", "--sq"])
    elif(args.op):
        print(f"OPNORM_{args.p} (1000):")
        if args.p != 0.5:
            subprocess.run(["python3", "-m", "compression.test_inference", "--op", "-p", str(args.p)])
        else:
            subprocess.run(["python3", "-m", "compression.test_inference", "--op"])
    elif(args.l1):
        print(f"L1_{args.p} (1000):")
        if args.p != 0.5:
            subprocess.run(["python3", "-m", "compression.test_inference", "--l1", "-p", str(args.p)])
        else:
            subprocess.run(["python3", "-m", "compression.test_inference", "--l1"])
    elif(args.comb):
        print("COMB (1000):")
        subprocess.run(["python3", "-m", "compression.test_inference", "--comb"])

    # End monitoring by terminating the bash script
    monitor_process.terminate()

    # Get the output of the bash script
    output = monitor_process.communicate()[0].decode('utf-8')

    # Extract average CPU and memory usage from the output
    lines = output.split('\n')[30:-30]
    total_cpu = 0
    total_mem = 0
    total_cpu_temp = 0
    count = 0
    for line in lines:
        try:
            if line.startswith("CPU Usage:"):
                total_cpu += float(line.split(':')[1].strip()[:-1])
                count += 1
            elif line.startswith("Memory Usage:"):
                total_mem += float(line.split(':')[1].strip()[:-3])
            elif line.startswith("CPU Temperature:"):
                total_cpu_temp += float(line.split(':')[1].strip()[:-3])
        except ValueError:
            print(line)
    
    average_cpu = total_cpu / count
    average_mem = total_mem / count
    average_cpu_temp = total_cpu_temp / count
    total_cpu_units = total_cpu

    print("Total CPU Usage during Python script execution:", total_cpu)
    print("Average CPU Usage during Python script execution:", average_cpu, "%")
    print("Average Memory Usage during Python script execution:", average_mem, "MB")
    print("Average CPU Temperature during Python script execution:", average_cpu_temp, "°C")

def run_all():
    run_all_help("Model_PANN (1000):", ["python3", "-m", "compression.test_inference", "--base"])
    time.sleep(60)
    run_all_help("PANN_QAT (1000):", ["python3", "-m", "compression.test_inference", "--qat"])
    time.sleep(60)
    run_all_help("PANN_QATv2 (1000):", ["python3", "-m", "compression.test_inference", "--qat2"])
    time.sleep(60)
    run_all_help("PANN_SQ (1000):", ["python3", "-m", "compression.test_inference", "--sq"])
    time.sleep(60)
    pruning = [0.5, 0.6 , 0.7, 0.81, 0.91]
    for p in pruning:
        run_all_help(f"OPNORM_{p} (1000):", ["python3", "-m", "compression.test_inference", "--op", "-p", str(p)])
        time.sleep(60)
    for p in pruning:
        run_all_help(f"L1_{p} (1000):", ["python3", "-m", "compression.test_inference", "--l1", "-p", str(p)])
        time.sleep(60)
    run_all_help("COMB (1000):", ["python3", "-m", "compression.test_inference", "--comb"])
    

def run_all_help(model, command):
    # Start monitoring CPU and memory usage
    monitor_process = subprocess.Popen(["./compression/scripts/monitor_usage.sh"], stdout=subprocess.PIPE)

    print(model, command)
    subprocess.run(command)

    # End monitoring by terminating the bash script
    monitor_process.terminate()

    # Get the output of the bash script
    output = monitor_process.communicate()[0].decode('utf-8')

    # Extract average CPU and memory usage from the output
    lines = output.split('\n')[30:-30]
    total_cpu = 0
    total_mem = 0
    total_cpu_temp = 0
    count = 0
    for line in lines:
        try:
            if line.startswith("CPU Usage:"):
                total_cpu += float(line.split(':')[1].strip()[:-1])
                count += 1
            elif line.startswith("Memory Usage:"):
                total_mem += float(line.split(':')[1].strip()[:-3])
            elif line.startswith("CPU Temperature:"):
                total_cpu_temp += float(line.split(':')[1].strip()[:-3])
        except ValueError:
            print(line)
    
    average_cpu = total_cpu / count
    average_mem = total_mem / count
    average_cpu_temp = total_cpu_temp / count
    total_cpu_units = total_cpu

    print("Total CPU Usage during Python script execution:", total_cpu)
    print("Average CPU Usage during Python script execution:", average_cpu, "%")
    print("Average Memory Usage during Python script execution:", average_mem, "MB")
    print("Average CPU Temperature during Python script execution:", average_cpu_temp, "°C")
    
def parser():
    parser = argparse.ArgumentParser(description="Argument parser for the provided variables")
    parser.add_argument("--base", default=False, action="store_true", help="Enable MODEL_PANN")
    parser.add_argument("--qat", default=False, action="store_true", help="Enable PANN_QAT")
    parser.add_argument("--qat2", default=False, action="store_true", help="Enable PANN_QAT_V2")
    parser.add_argument("--sq", default=False, action="store_true", help="Enable PANN_SQ")
    parser.add_argument("--op", default=False, action="store_true", help="Enable OPNORM_PRUNING")
    parser.add_argument("-p", type=float, default=0.5, help="Value of P if pruning is enabled (default: 0.5)")
    parser.add_argument("--l1", default=False, action="store_true", help="Enable L1_PRUNING")
    parser.add_argument("--comb", default=False, action="store_true", help="Enable COMBINATION")
    parser.add_argument("--all", default=False, action="store_true", help="Enable ALL models")

    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    if not args.all:
        run_python_script(args)
    else:
        run_all()

