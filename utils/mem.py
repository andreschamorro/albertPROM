import re

def get_memory_peak():
    pattern = re.compile("VmPeak")
    with open("/proc/self/status",'r') as status:
        for line in status:
            for match in re.finditer(pattern, line):
                return line
