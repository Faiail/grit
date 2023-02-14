import itertools
import sys

import torchvision.transforms

sys.path.append("..")
import os
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import torch
from models.common.attention import MemoryAttention
from models.caption.detector import build_detector
from models.caption import Transformer, GridFeatureNetwork, CaptionGenerator
from datasets.caption.field import TextField
from datasets.caption.transforms import get_transform
from engine.utils import nested_tensor_from_tensor_list
from ArtGraphDataset import ArtGraphDataset
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.caption import metrics
import json


device = torch.device(f"cuda:0")


def get_model(checkpoint_path, config):
    detector = build_detector(config).to(device)

    grit_net = GridFeatureNetwork(
        pad_idx=config.model.pad_idx,
        d_in=config.model.grid_feat_dim,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        attention_module=MemoryAttention,
        **config.model.grit_net,
    )
    cap_generator = CaptionGenerator(
        vocab_size=config.model.vocab_size,
        max_len=config.model.max_len,
        pad_idx=config.model.pad_idx,
        cfg=config.model.cap_generator,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        **config.model.cap_generator,
    )

    model = Transformer(
        grit_net,
        cap_generator,
        detector=detector,
        use_gri_feat=config.model.use_gri_feat,
        use_reg_feat=config.model.use_reg_feat,
        config=config,
    )
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def eval(dataloader, model, text_field):
    with tqdm(desc=f'Epoch {1} - evaluation on {"artgraph"}', unit='it', total=len(dataloader)) as pbar:
        model.eval()
        gen, gts = {}, {}
        for it, (images, caps_gt) in enumerate(dataloader):
            images = nested_tensor_from_tensor_list([a for a in images])
            images = images.to(device)
            caps_gt = list(map(lambda x: x.split(' '), caps_gt))
            with torch.no_grad():
                out, _ = model(
                    images,
                    seq=None,
                    use_beam_search=True,
                    max_len=config.model.beam_len,
                    eos_idx=config.model.eos_idx,
                    beam_size=config.model.beam_size,
                    out_size=1,
                    return_probs=False,
                )
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen[f'{it}_{i}'] = [gen_i]
                gts[f'{it}_{i}'] = gts_i
            pbar.update()

    print('computing scores')
    gts = metrics.PTBTokenizer.tokenize(gts)
    gen = metrics.PTBTokenizer.tokenize(gen)
    scores, _ = metrics.compute_scores(gts, gen)
    return scores


if __name__ == '__main__':
    checkpoint = "../grit_checkpoint_4ds.pth"
    vocab_path = 'vocab.json'
    GlobalHydra.instance().clear()
    initialize(config_path="../configs/caption")
    config = compose(config_name='artgraph_config.yaml', overrides=[f"exp.checkpoint={checkpoint}"])

    model = get_model(checkpoint, config)

    # get base transform
    transform = get_transform(config.dataset.transform_cfg)['valid']
    transform.transforms.append(torchvision.transforms.Resize((224, 224)))

    text_field = TextField(vocab_path=vocab_path)
    data = pd.read_csv('artgraph_captions.csv', index_col=0)
    dataset = ArtGraphDataset(data=data, root='./images-resized', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    scores = eval(dataloader, model, text_field)
    json.dump(scores, open('artgraph_scores.json', mode='w+'))

