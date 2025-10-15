from os.path import join
import numpy as np
import torch
import clip
import hydra
from tqdm import tqdm
from PIL import Image
from termcolor import cprint


@hydra.main(config_path="../../configs/thingseeg2", config_name="clip")
def run(args):
    model, preprocess = clip.load(args.vision.pretrained_model)
    model = model.eval().requires_grad_(False).to("cuda")
    
    metadata = np.load(join(args.data_dir, "image_metadata.npy"), allow_pickle=True)[()]
    
    for train in [False, True]:
        image_paths = metadata[f"{'train' if train else 'test'}_img_files"]
        concepts = metadata[f"{'train' if train else 'test'}_img_concepts"]

        image_embeds = []
        for image_path, concept in tqdm(
            zip(image_paths, concepts), total=len(image_paths), desc="Preproc & embedding images"
        ):
            image_path = join(
                args.data_dir,
                f"{'training' if train else 'test'}_images",
                concept,
                image_path,
            )
            image = preprocess(Image.open(image_path).convert("RGB")) # ( c, h, w )
            image_embed = model.encode_image(image.unsqueeze(0).to("cuda")).float().cpu()
            image_embeds.append(image_embed)
            
        image_embeds = torch.cat(image_embeds)
        cprint(image_embeds.shape, "green")
        
        torch.save(
            image_embeds,
            join(args.data_dir, f"{'training' if train else 'test'}_image_embs.pt"),
        )


if __name__ == "__main__":
    run()