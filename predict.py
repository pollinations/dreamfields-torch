# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List
import torch
import argparse

from slugify import slugify

from nerf.provider import NeRFDataset
from nerf.utils import *
from nerf.network import NeRFNetwork
import pymeshlab
from time import sleep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _print(stuff):
    print(stuff)
    print(os.popen("ls /tmp").read())


class Predictor(BasePredictor):
    def setup(self):
        """Load the self.model into memory to make running multiple predictions efficient"""
        self.model = NeRFNetwork(
            bound=1,
            cuda_ray=True,
            density_scale=1,
        )

    def predict(
        self,
        text: str = Input(description="Text to render"),
        image: Path = Input(description="ref image prompt", default=None),
        seed: int = Input(description="random seed", default=123),
        iters: int = Input(description="training iters", default=3000),
        lr: float = Input(description="initital learning rate", default=5e-4),
        num_rays: int = Input(description="number of rays", default=4096),
        # cuda_ray: bool = Input(description="use CUDA raymarching instead of pytorch", default=False),
        num_steps: int = Input(description="num steps sampled per ray (only valid when not using --cuda_ray)", default=512),
        upsample_steps: int = Input(description="num steps up-sampled per ray (only valid when not using --cuda_ray)", default=0),
        max_ray_batch: int = Input(description="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)", default=4096),
        fp16: bool = Input(description="use amp mixed precision training", default=False),
        # cc: bool = Input(description="use TensoRF", default=False),
        # bound: float = Input(description="assume the scene is bounded in box(-bound, bound)", default=1),
        dt_gamma: float = Input(description="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)", default=0),
        w: int = Input(description="render width for CLIP training (<=224)", default=128),
        h: int = Input(description="render height for CLIP training (<=224)", default=128),
        tau_0: float = Input(description="target mean transparency 0", default=0.5),
        tau_1: float = Input(description="target mean transparency 1", default=0.8),
        tau_step: float = Input(description="steps to anneal from tau_0 to tau_1", default=500),
        aug_copy: int = Input(description="augmentation copy for each renderred image before feeding into CLIP", default=8),
        dir_text: bool = Input(description="direction encoded text prompt", default=False),
        radius: float = Input(description="default GUI camera radius from center", default=3),
        W: int = Input(description="W", default=800),
        H: int = Input(description="H", default=800),
        fovy: float = Input(description="default GUI camera fovy", default=90),
        max_spp: int = Input(description="GUI rendering max sample per pixel", default=64)
    ) -> Path:
        """Run a single prediction on the self.model"""
        opt = argparse.Namespace(
            text=text,
            image=image,
            seed=seed,
            iters=iters,
            lr=lr,
            num_rays=num_rays,
            # cuda_ray=cuda_ray,
            cuda_ray=True,
            num_steps=num_steps,
            upsample_steps=upsample_steps,
            max_ray_batch=max_ray_batch,
            fp16=fp16,
            # cc=cc,
            # bound=bound,
            dt_gamma=dt_gamma,
            w=w,
            h=h,
            tau_0=tau_0,
            tau_1=tau_1,
            tau_step=tau_step,
            aug_copy=aug_copy,
            dir_text=dir_text,
            radius=radius,
            W=W,
            H=H,
            fovy=fovy,
            max_spp=max_spp,
            workspace="/outputs",
            ckpt="latest"
        )
        if not os.path.exists(opt.workspace):
            os.makedirs(opt.workspace)
        seed_everything(opt.seed)
        model = self.model
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, radius=opt.radius, fovy=opt.fovy, size=100).dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=20)

        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, radius=opt.radius, fovy=opt.fovy, size=10).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, radius=opt.radius, fovy=opt.fovy, size=10).dataloader()
        trainer.test(test_loader)

        target_path = os.path.join("/tmp", f"y_{slugify(text)}.obj")
        trainer.save_mesh(target_path)
        
        # it seems that save_mesh may sometimes not finish writing the mesh file? or is it zero bytes?
        # so we wait a bit and then check if the file is there, but maximum 60 seconds
        for i in range(60):
            if os.path.getsize(target_path) > 100:
                break
            time.sleep(20)
        
        ms = pymeshlab.MeshSet()

        ms.load_new_mesh(target_path)
        
        # run filter meshing_invert_face_orientation
        ms.meshing_invert_face_orientation()
        ms.compute_color_transfer_vertex_to_face()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=6500)
        ms.save_current_mesh(target_path) 

        text_slug = slugify(text)



        os.system(f"cp -v {target_path} /outputs")
        os.system(f"ls -l {target_path}")

        # save as glb file
        target_glb_path = os.path.join("/outputs",f"z_{text_slug}.glb")
        print("running ", f"obj2gltf -i {target_path} -o {target_glb_path}")
        
        os.system(f"obj2gltf -i {target_path} -o {target_glb_path}")

        return Path(target_path)

