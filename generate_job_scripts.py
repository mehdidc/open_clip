LOGS = "logs"
template = open("template.lsf").read()
for dataset in ("laion400m",):
    for model in ("ViT-L/14",):
        for nodes in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            gpus = (1, 6) if nodes == 1 else (6,)
            for gpus_per_node in gpus:
                model_slug = model.replace("/", "-")
                batch_size = 256 if model == "ViT-B/32" else 64
                extra = "--synthetic-data" if dataset == "synthetic" else ""
                name = f"N{nodes}_G{gpus_per_node}_M{model_slug}_B{batch_size}_{dataset}"
                s = template
                s = s.replace("TEMPLATE_NODES", str(nodes))
                s = s.replace("TEMPLATE_GPUS_PER_NODE", str(gpus_per_node))
                s = s.replace("TEMPLATE_EXTRA", extra)
                s = s.replace("TEMPLATE_BATCH_SIZE", str(batch_size))
                s = s.replace("TEMPLATE_MODEL", model)
                s = s.replace("TEMPLATE_NAME", name)
                s = s.replace("TEMPLATE_LOGS", LOGS)
                with open("job_" + name + ".lsf", "w") as fd:
                    fd.write(s)
