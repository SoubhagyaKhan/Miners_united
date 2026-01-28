import matplotlib.pyplot as plt
import sys

data = sys.argv[1]
out_prefix = sys.argv[2]

TIMEOUT = 7200 

supports, ap, fp = [], [], []

with open(data) as f:
    next(f)
    for line in f:
        s, a, fpt = line.strip().split(",")

        supports.append(int(s))

        ap.append(TIMEOUT if a == "timeout" else float(a))
        fp.append(TIMEOUT if fpt == "timeout" else float(fpt))

# Linear scale plot
plt.figure()
plt.plot(supports, ap, marker="o", label="Apriori")
plt.plot(supports, fp, marker="o", label="FP-Growth")
plt.xlabel("Support Threshold (%)")
plt.ylabel("Runtime (seconds)")
plt.legend()
plt.grid(True)
plt.savefig(out_prefix + "_linear.png")
plt.close()

# # Log scale plots
# plt.figure()
# plt.plot(supports, ap, marker="o", label="Apriori")
# plt.plot(supports, fp, marker="o", label="FP-Growth")
# plt.xlabel("Support Threshold (%)")
# plt.ylabel("Runtime (seconds, log scale)")
# plt.yscale("log")
# plt.legend()
# plt.grid(True, which="both")
# plt.savefig(out_prefix + "_log.png")
# plt.close()