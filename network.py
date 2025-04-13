import psutil
print(psutil.net_if_stats())

import subprocess
result = subprocess.run(['iwconfig'], capture_output=True, text=True)
print(result)