from glob import glob
import os
import re
LOGS = "/gpfs/alpine/csc499/scratch/mcherti/open_clip/logs"
pattern = r"Throughput: ([0-9]+\.[0-9]+)"
out = open("results.csv", "w")
for logfile in glob(os.path.join(LOGS, "**", "out.log")):
    name = os.path.basename(os.path.dirname(logfile))
    nodes, gpus_per_node, model, bs, data = name.split("_")
    nodes = nodes[1:]
    gpus_per_node = gpus_per_node[1:]
    model = model[1:]
    bs = bs[1:]
    with open(logfile, "r") as fd:
        data = fd.read()
    res = re.search(pattern, data)
    if res:
        out.write(f"{nodes},{gpus_per_node},{model},{bs},{res.groups(1)[0]}\n")
out.close()
