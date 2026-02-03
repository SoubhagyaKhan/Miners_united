import subprocess
import time
import os
import sys

apriori = sys.argv[1]
fpgrowth = sys.argv[2]
dataset = sys.argv[3]
outdir = sys.argv[4]

supports = [5, 10, 25, 50, 90]

os.makedirs(outdir, exist_ok=True)

TIMEOUT = 7200 # FOR QUESTION 2
# 7200 for QUESTION 1

# def run(cmd, ap_out):
#     start = time.time()
#     with open(f"{ap_out}/log.txt", "w") as log:
#         subprocess.run(cmd, shell=True, stdout=log, stderr=log)
#     return time.time() - start

# adding time out to handle apirori at 5% support threshold
def run(cmd, out_dir, timeout=None):
    start = time.time()
    try:
        with open(f"{out_dir}/log.txt", "w") as log:
            subprocess.run(
                cmd,
                shell=True,
                stdout=log,
                stderr=log,
                timeout=timeout
            )
        return time.time() - start, "ok"
    except subprocess.TimeoutExpired:
        return timeout, "timeout"

ap_times, fp_times = [], []

for s in supports:
    print(f"\nSupport threshold: {s}%")

    ap_out = os.path.join(outdir, f"ap{s}")
    fp_out = os.path.join(outdir, f"fp{s}")

    os.makedirs(ap_out, exist_ok=True)
    os.makedirs(fp_out, exist_ok=True)

    timeout = TIMEOUT if s == 5 else None # 2-hour timeout chosen to mark infeasible Apriori runs at low support

    print("Running Apriori...")

    ap_cmd = f"{apriori} -s{s} {dataset} {ap_out}/out.txt"
    ap_time, ap_status = run(ap_cmd, ap_out, timeout = timeout)
    
    if ap_status == "timeout":
        print(f"Support threshold : {s} :: Apriori timed out in {ap_time:.2f} seconds")
    else:
        print(f"Support threshold : {s} : Apriori done in {ap_time:.2f} seconds")
    print("Running FP-Growth...")

    fp_cmd = f"{fpgrowth} -s{s} {dataset} {fp_out}/out.txt"
    fp_time, fp_status = run(fp_cmd, fp_out, timeout = timeout)

    if fp_status == "timeout":
        print(f"Support threshold : {s} : fp_growth timed out in {fp_time:.2f} seconds")
    else:
        print(f"Support threshold : {s} : FP-Growth done in {fp_time:.2f} seconds")

    ap_times.append(ap_time)
    fp_times.append(fp_time)

with open(os.path.join(outdir, "times.txt"), "w") as f:
    f.write("support,apriori,fp\n")
    for s,a,fpt in zip(supports, ap_times, fp_times):
        a_val = "timeout" if a == TIMEOUT else f"{a:.2f}"
        fpt_val = "timeout" if fpt == TIMEOUT else f"{fpt:.2f}"
        f.write(f"{s},{a_val},{fpt_val}\n")

# bash q1_1.sh $(pwd)/apriori/apriori/src/apriori $(pwd)/fpgrowth/fpgrowth/src/fpgrowth $(pwd)/dataset/webdocs.dat $(pwd)/output_task1 > output_task1/run.log 2>&1