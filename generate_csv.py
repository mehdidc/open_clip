from glob import glob
import os
import re
import pandas as pd
LOGS = "logs"
pattern_images_per_sec = r"Throughput: ([0-9]+\.[0-9]+)"
pattern_secs = r"Benchmarking completed in ([0-9]+\.[0-9]+) seconds"
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
    imps = re.search(pattern_images_per_sec, data)
    secs = re.search(pattern_secs, data)
    if imps and secs:
        rows.append({
            "nodes": int(nodes),
            "gpus_per_node": int(gpus_per_node),
            "model": model,
            "local_batch_size": int(bs),
            "samples_per_sec": float(imps.groups(1)[0]),
            "secs_per_batch": float(secs.groups(1)[0]) / (1000),
        })
df = pd.DataFrame(rows)
df = df.sort_values(by=["nodes", "gpus_per_node"])
df.to_csv("results.csv", index=False)
