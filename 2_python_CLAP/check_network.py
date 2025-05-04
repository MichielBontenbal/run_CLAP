import subprocess
result = subprocess.run(['iwconfig'], capture_output=True, text=True)
output = result.stdout
#print(output)

print(50*'-')
for line in output.splitlines():
    if 'Signal level' in line:
        print(line.strip())
    if 'ESSID' in line:
        print(line.strip())
    else:
        pass
print(50*'-')