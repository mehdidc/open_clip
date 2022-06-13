from glob import glob
import os
import re
import pandas as pd
LOGS = "logs"
pattern = r"Throughput: ([0-9]+\.[0-9]+)"
rows  = []
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
        rows.append({
            "nodes": int(nodes),
            "gpus_per_node": int(gpus_per_node),
            "model": model,
            "local_batch_size": int(bs),
            "samples_per_sec": float(res.groups(1)[0]),
        })
df = pd.DataFrame(rows)
df = df.sort_values(by=["nodes", "gpus_per_node"])
df.to_csv("results.csv", index=False)
