# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
sys.path.append('../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from ibrnet.data_loaders.create_training_dataset import create_training_dataset
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import tensorflow as tf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # Create IBRNet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    testsavedir = os.path.join(args.output_dir, os.path.basename(args.datadir))
    os.makedirs(testsavedir, exist_ok=True)

    test_dataset, test_sampler = create_training_dataset(args, mode='val')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               shuffle=False)

    for i, data in enumerate(test_loader):
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        img_file = data['img_file'][0]

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

            ret = render_single_image(ray_sampler=ray_sampler,
                                    ray_batch=ray_batch,
                                    model=model,
                                    projector=projector,
                                    chunk_size=args.chunk_size,
                                    det=True,
                                    N_samples=args.N_samples,
                                    inv_uniform=args.inv_uniform,
                                    N_importance=args.N_importance,
                                    white_bkgd=args.white_bkgd,
                                    featmaps=featmaps)

            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(testsavedir, img_file), fine_pred_rgb)

