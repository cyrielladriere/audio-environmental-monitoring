import subprocess

def run_python_script():
    # Start monitoring CPU and memory usage
    monitor_process = subprocess.Popen(["./compression/scripts/monitor_usage.sh"], stdout=subprocess.PIPE)

    subprocess.run(["python3", "-m", "compression.test_inference"])

    # End monitoring by terminating the bash script
    monitor_process.terminate()

    # Get the output of the bash script
    output = monitor_process.communicate()[0].decode('utf-8')

    # Extract average CPU and memory usage from the output
    lines = output.split('\n')[60:-30]
    total_cpu = 0
    total_mem = 0
    count = 0
    for line in lines:
        if line.startswith("CPU Usage:"):
            total_cpu += float(line.split(':')[1].strip()[:-1])
            count += 1
        elif line.startswith("Memory Usage:"):
            total_mem += float(line.split(':')[1].strip()[:-3])
    
    average_cpu = total_cpu / count
    average_mem = total_mem / count

    print("Average CPU Usage during Python script execution:", average_cpu, "%")
    print("Average Memory Usage during Python script execution:", average_mem, "MB")

run_python_script()
