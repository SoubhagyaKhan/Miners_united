import sys
import matplotlib.pyplot as plt
import json

json_file = sys.argv[1]
out = sys.argv[2]

def parse(x):
    try:
        return float(x)
    except:
        return None

with open(json_file, "r") as f:
    data = json.load(f)

supports = [5, 10, 25, 50, 95]
gspan, fsg, gaston = [], [], []

print(data["gspan"])
print(data["fsg"])
print(data["gaston"])

for support in supports:
    gspan.append(parse(data["gspan"][str(support)]))
    fsg.append(parse(data["fsg"][str(support)]))
    gaston.append(parse(data["gaston"][str(support)]))

plt.figure(figsize=(8,6))

def plot_valid(x, y, label, marker):
    xs, ys = [], []
    for a,b in zip(x,y):
        if b is not None:
            xs.append(a)
            ys.append(b)
    plt.plot(xs, ys, marker=marker, label=label, linewidth=2)

plot_valid(supports, gspan, "gSpan", "o")
plot_valid(supports, fsg, "FSG", "s")
plot_valid(supports, gaston, "Gaston", "^")

plt.xlabel("Minimum Support (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison: gSpan vs FSG vs Gaston")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out, dpi=300)
