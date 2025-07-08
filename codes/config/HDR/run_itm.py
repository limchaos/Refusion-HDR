import argparse
import sys
import os
import numpy as np
import torch

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util

def main():
    parser = argparse.ArgumentParser(description="Run Refusion HDR pipeline.")
    parser.add_argument(
        '-opt_path', 
        type=str, 
        default='options/test/refusion.yml', 
        help='Path to options YAML file'
    )
    parser.add_argument(
        '-input_dir', 
        type=str, 
        default='AIM_TEST/', 
        help='Directory with input images'
    )
    parser.add_argument(
        '-output_dir', 
        type=str, 
        default='output/', 
        help='Directory to save output results'
    )
    parser.add_argument(
        '-pretrained_path', 
        type=str, 
        default='lastest_EMA.pth', 
        help='Path to pretrained model'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load options
    opt = option.parse(args.opt_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    opt['path']['pretrain_model_G'] = args.pretrained_path

    print(opt)

    # load pretrained model by default
    model = create_model(opt)
    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"], 
        T=opt["sde"]["T"], 
        schedule=opt["sde"]["schedule"], 
        eps=opt["sde"]["eps"], 
        device=model.device
    )
    sde.set_model(model.model)

    PU21 = util.PU21Encoder() 

    for filename in os.listdir(args.input_dir):
        path = os.path.join(args.input_dir, filename)
        ldr = util.read_img(path) 
        # run SDE
        ldr_tensor = torch.tensor(ldr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        noisy_tensor = sde.noise_state(ldr_tensor)
        model.feed_data(noisy_tensor, ldr_tensor)
        model.test(sde)
        visuals = model.get_current_visuals(need_GT=False)
        hdr = util.tensor2img(visuals["Output"].squeeze(), out_type=np.float32)
        # decode to linear HDR
        hdr = PU21.decode(hdr * PU21.encode(1000))
        hdr = hdr.astype(np.float32)
        # save
        dst_path = os.path.join(args.output_dir, filename)[:-3]
        np.save(dst_path, hdr)

if __name__ == "__main__":
    main()

