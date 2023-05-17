import os
import glob
import json
from operator import is_
import numpy as np
from termcolor import cprint
from PIL import Image


def export_gif(path, example: np.ndarray) -> None:
    example = example[:10].reshape(-1, example.shape[-2], example.shape[-1])
    gif = [Image.fromarray(p) for p in example]
    gif[0].save(
        path,
        save_all=True,
        append_images=gif[1:],
        duration=100,
        loop=0,
    )


# NOTE currently only works for gwilliams2022.yml
def check_preprocs(args, data_dir):
    is_processed = False
    preproc_dirs = glob.glob(data_dir + "*/")

    for preproc_dir in preproc_dirs:
        try:
            with open(preproc_dir + "settings.json") as f:
                settings = json.load(f)

            x_done = settings.pop("x_done") if "x_done" in settings else False
            y_done = settings.pop("y_done") if "y_done" in settings else False
        except:
            cprint("No settings.json under preproc dir", color="yellow")
            continue

        try:
            is_processed = np.all([v == args.preprocs[k] for k, v in settings.items()])
            if is_processed:
                cprint(f"Using preprocessing {preproc_dir}", color="cyan")
                break
        except:
            cprint("Preproc hyperparameter name mismatch", color="yellow")
            continue

    if not is_processed:
        preproc_dir = data_dir + str(len(preproc_dirs)) + "/"
        os.mkdir(preproc_dir)

        args.preprocs.update({"x_done": False, "y_done": False})

        with open(preproc_dir + "settings.json", 'w') as f:
            json.dump(args.preprocs, f)

    else:
        args.preprocs.update({"x_done": x_done, "y_done": y_done})

    return args, preproc_dir