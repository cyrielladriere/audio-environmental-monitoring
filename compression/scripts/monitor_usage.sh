#!/bin/bash
# This script monitors CPU and memory usage

while :
do 
  # Get the current usage of CPU and memory
  cpuUsage=$(top -bn1 | awk '/Cpu/ { print $2}')
  memUsage=$(free -m | awk '/Mem/{print $3}')
  cpuTemp=$(/usr/bin/vcgencmd measure_temp | awk -F= '/temp/{print $2}')
  temp="${cpuTemp%%\'*}"

  # Print the usage
  echo "CPU Usage: $cpuUsage%"
  echo "Memory Usage: $memUsage MB"
  echo "CPU Temperature: $temp °C"
 
  # Sleep for 1 second
  sleep 1
done