import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding
from mld.transforms.multi_scale_init import *
from .base import BaseModel


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.joint_type = cfg.DATASET.JOINT_TYPE
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        if self.nfeats == 263:
            self.init_s1_to_s2 = humanml_init_s1_to_s2()
            self.init_s1_to_s3 = humanml_init_s1_to_s3()
            self.init_s1_to_s4 = humanml_init_s1_to_s4()
        elif self.nfeats == 251:
            self.init_s1_to_s2 = kit_init_s1_to_s2()
            self.init_s1_to_s3 = kit_init_s1_to_s3()
            self.init_s1_to_s4 = kit_init_s1_to_s4()
        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        if self.vae_type != "no":
            self.vae_s1 = instantiate_from_config(cfg.model.motion_vae)
            self.vae_s2 = instantiate_from_config(cfg.model.motion_vae)
            self.vae_s3 = instantiate_from_config(cfg.model.motion_vae)
            self.vae_s4 = instantiate_from_config(cfg.model.motion_vae)
        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert","actor"]:
                self.vae_s1.training = False
                for p in self.vae_s1.parameters():
                    p.requires_grad = False
                for p in self.vae_s2.parameters():
                    p.requires_grad = False
                for p in self.vae_s3.parameters():
                    p.requires_grad = False
                for p in self.vae_s4.parameters():
                    p.requires_grad = False

            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        self.denoiser_s1 = instantiate_from_config(cfg.model.denoiser)
        self.denoiser_s2 = instantiate_from_config(cfg.model.denoiser)
        self.denoiser_s3 = instantiate_from_config(cfg.model.denoiser)
        self.denoiser_s4 = instantiate_from_config(cfg.model.denoiser)

        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler_s1 = instantiate_from_config(cfg.model.scheduler)
        self.scheduler_s2 = instantiate_from_config(cfg.model.scheduler)
        self.scheduler_s3 = instantiate_from_config(cfg.model.scheduler)
        self.scheduler_s4 = instantiate_from_config(cfg.model.scheduler)

        self.noise_scheduler_s1 = instantiate_from_config(
            cfg.model.noise_scheduler)
        self.noise_scheduler_s2 = instantiate_from_config(
            cfg.model.noise_scheduler)
        self.noise_scheduler_s3 = instantiate_from_config(
            cfg.model.noise_scheduler)
        self.noise_scheduler_s4 = instantiate_from_config(
            cfg.model.noise_scheduler)

        if self.condition in ["text", "text_uncond"]:
            self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond']:
            self.feats2joints = datamodule.feats2joints
        elif self.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.DATASET.SMPL_PATH)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "text_mot_match/model/finest.tar"), map_location='cuda:0')
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        texts = batch["text"]
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)


        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale='s1', coarse_motion=None, reverse_in_test_flage=False):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance and reverse_in_test_flage:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        if scale == 's1':
            latents = latents * self.scheduler_s1.init_noise_sigma
            # set timesteps
            self.scheduler_s1.set_timesteps(
                self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler_s1.timesteps.to(encoder_hidden_states.device)
        elif scale == 's2':
            latents = latents * self.scheduler_s2.init_noise_sigma
            # set timesteps
            self.scheduler_s2.set_timesteps(
                self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler_s2.timesteps.to(encoder_hidden_states.device)
        elif scale == 's3':
            latents = latents * self.scheduler_s3.init_noise_sigma
            # set timesteps
            self.scheduler_s3.set_timesteps(
                self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler_s3.timesteps.to(encoder_hidden_states.device)
        elif scale == 's4':
            latents = latents * self.scheduler_s4.init_noise_sigma
            # set timesteps
            self.scheduler_s4.set_timesteps(
                self.cfg.model.scheduler.num_inference_timesteps)
            timesteps = self.scheduler_s4.timesteps.to(encoder_hidden_states.device)

        extra_step_kwargs = {}
        if scale == 's1':
            if "eta" in set(
                    inspect.signature(self.scheduler_s1.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        if scale == 's2':
            if "eta" in set(
                    inspect.signature(self.scheduler_s2.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        if scale == 's3':
            if "eta" in set(
                    inspect.signature(self.scheduler_s3.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        if scale == 's4':
            if "eta" in set(
                    inspect.signature(self.scheduler_s4.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta


        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance and reverse_in_test_flage else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance and reverse_in_test_flage
                               else lengths)

            if scale == 's1':
                noise_pred = self.denoiser_s1(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    coarse_motion = coarse_motion,
                    lengths=lengths_reverse,
                )[0]
            elif scale == 's2':
                noise_pred = self.denoiser_s2(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    coarse_motion = coarse_motion,
                    lengths=lengths_reverse,
                )[0]
            elif scale == 's3':
                noise_pred = self.denoiser_s3(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    coarse_motion = coarse_motion,
                    lengths=lengths_reverse,
                )[0]
            elif scale == 's4':
                noise_pred = self.denoiser_s4(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    coarse_motion = None,
                    lengths=lengths_reverse,
                )[0]

            # perform guidance
            if self.do_classifier_free_guidance and reverse_in_test_flage:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            if scale == 's1':
                latents = self.scheduler_s1.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            elif scale == 's2':
                latents = self.scheduler_s2.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            elif scale == 's3':
                latents = self.scheduler_s3.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            elif scale == 's4':
                latents = self.scheduler_s4.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None, scale='s1', coarse_motion=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """

        latents = latents.permute(1, 0, 2)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps_s1 = torch.randint(
            0,
            self.noise_scheduler_s1.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps_s1 = timesteps_s1.long()

        timesteps_s2 = torch.randint(
            0,
            self.noise_scheduler_s2.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps_s2 = timesteps_s2.long()

        timesteps_s3 = torch.randint(
            0,
            self.noise_scheduler_s3.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps_s3 = timesteps_s3.long()

        timesteps_s4 = torch.randint(
            0,
            self.noise_scheduler_s4.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps_s4 = timesteps_s4.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        if scale == 's1':
            noisy_latents = self.noise_scheduler_s1.add_noise(latents.clone(), noise, timesteps_s1)
            noise_pred = self.denoiser_s1(
                sample=noisy_latents,
                timestep=timesteps_s1,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                coarse_motion=coarse_motion,
                return_dict=False,
            )[0]
        elif scale == 's2':
            noisy_latents = self.noise_scheduler_s2.add_noise(latents.clone(), noise, timesteps_s2)
            noise_pred = self.denoiser_s2(
                sample=noisy_latents,
                timestep=timesteps_s2,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                coarse_motion = coarse_motion,
                return_dict=False,
            )[0]
        elif scale == 's3':
            noisy_latents = self.noise_scheduler_s3.add_noise(latents.clone(), noise, timesteps_s3)
            noise_pred = self.denoiser_s3(
                sample=noisy_latents,
                timestep=timesteps_s3,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                coarse_motion = coarse_motion,
                return_dict=False,
            )[0]
        elif scale == 's4':
            noisy_latents = self.noise_scheduler_s4.add_noise(latents.clone(), noise, timesteps_s4)
            noise_pred = self.denoiser_s4(
                sample=noisy_latents,
                timestep=timesteps_s4,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
                coarse_motion = None,
                return_dict=False,
            )[0]

        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref_s1 = batch["motion"]
        lengths = batch["length"]

        joints_ref_s1 = self.feats2joints(feats_ref_s1)
        joints_ref_s2 = self.init_s1_to_s2(joints_ref_s1)
        joints_ref_s3 = self.init_s1_to_s3(joints_ref_s1)
        joints_ref_s4 = self.init_s1_to_s4(joints_ref_s1)

        if self.vae_type in ["mld", "vposert", "actor"]:

            motion_z_s1, dist_m_s1 = self.vae_s1.encode(feats_ref_s1, lengths)
            feats_rst_s1 = self.vae_s1.decode(motion_z_s1, lengths, 's1')


            motion_z_s2, dist_m_s2 = self.vae_s2.encode(joints_ref_s2, lengths)
            joints_rst_s2 = self.vae_s2.decode(motion_z_s2, lengths, 's2')

            motion_z_s3, dist_m_s3 = self.vae_s3.encode(joints_ref_s3, lengths)
            joints_rst_s3 = self.vae_s3.decode(motion_z_s3, lengths, 's3')

            motion_z_s4, dist_m_s4 = self.vae_s4.encode(joints_ref_s4, lengths)
            joints_rst_s4 = self.vae_s4.decode(motion_z_s4, lengths, 's4')
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z_s1, dist_rm_s1 = self.vae_s1.encode(feats_rst_s1, lengths)
        recons_z_s2, dist_rm_s2 = self.vae_s2.encode(joints_rst_s2, lengths)
        recons_z_s3, dist_rm_s3 = self.vae_s3.encode(joints_rst_s3, lengths)
        recons_z_s4, dist_rm_s4 = self.vae_s4.encode(joints_rst_s4, lengths)

        # joints recover
        if self.condition == "text":
            joints_rst_s1 = self.feats2joints(feats_rst_s1)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst_s1, mask)

        if dist_m_s1 is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref_s1 = torch.zeros_like(dist_m_s1.loc)
                scale_ref_s1 = torch.ones_like(dist_m_s1.scale)
                dist_ref_s1 = torch.distributions.Normal(mu_ref_s1, scale_ref_s1)
            else:
                dist_ref_s1 = dist_m_s1

        if dist_m_s2 is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref_s2 = torch.zeros_like(dist_m_s2.loc)
                scale_ref_s2 = torch.ones_like(dist_m_s2.scale)
                dist_ref_s2 = torch.distributions.Normal(mu_ref_s2, scale_ref_s2)
            else:
                dist_ref_s2 = dist_m_s2

        if dist_m_s3 is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref_s3 = torch.zeros_like(dist_m_s3.loc)
                scale_ref_s3 = torch.ones_like(dist_m_s3.scale)
                dist_ref_s3 = torch.distributions.Normal(mu_ref_s3, scale_ref_s3)
            else:
                dist_ref_s3 = dist_m_s3

        if dist_m_s4 is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref_s4 = torch.zeros_like(dist_m_s4.loc)
                scale_ref_s4 = torch.ones_like(dist_m_s4.scale)
                dist_ref_s4 = torch.distributions.Normal(mu_ref_s4, scale_ref_s4)
            else:
                dist_ref_s4 = dist_m_s4

        # cut longer part over max length
        min_len = min(feats_ref_s1.shape[1], feats_rst_s1.shape[1])
        rs_set = {
            # loss for scale s1
            "m_ref_s1": feats_ref_s1[:, :min_len, :],
            "m_rst_s1": feats_rst_s1[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m_s1": motion_z_s1.permute(1, 0, 2),
            "lat_rm_s1": recons_z_s1.permute(1, 0, 2),
            "joints_ref_s1": joints_ref_s1,
            "joints_rst_s1": joints_rst_s1,
            "dist_m_s1": dist_m_s1,
            "dist_ref_s1": dist_ref_s1,
            # loss for scale s2
            "lat_m_s2": motion_z_s2.permute(1, 0, 2),
            "lat_rm_s2": recons_z_s2.permute(1, 0, 2),
            "joints_ref_s2": joints_ref_s2,
            "joints_rst_s2": joints_rst_s2,
            "dist_m_s2": dist_m_s2,
            "dist_ref_s2": dist_ref_s2,
            # loss for scale s3
            "lat_m_s3": motion_z_s3.permute(1, 0, 2),
            "lat_rm_s3": recons_z_s3.permute(1, 0, 2),
            "joints_ref_s3": joints_ref_s3,
            "joints_rst_s3": joints_rst_s3,
            "dist_m_s3": dist_m_s3,
            "dist_ref_s3": dist_ref_s3,
            # loss for scale s4
            "lat_m_s4": motion_z_s4.permute(1, 0, 2),
            "lat_rm_s4": recons_z_s4.permute(1, 0, 2),
            "joints_ref_s4": joints_ref_s4,
            "joints_rst_s4": joints_rst_s4,
            "dist_m_s4": dist_m_s4,
            "dist_ref_s4": dist_ref_s4,

        }
        return rs_set

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"]
        joint_ref_s1 = self.feats2joints(feats_ref)
        joint_ref_s2 = self.init_s1_to_s2(joint_ref_s1)
        joint_ref_s3 = self.init_s1_to_s3(joint_ref_s1)
        joint_ref_s4 = self.init_s1_to_s4(joint_ref_s1)
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z_s1, dist_s1 = self.vae_s1.encode(feats_ref, lengths)
                z_s2, dist_s2 = self.vae_s2.encode(joint_ref_s2, lengths)
                z_s3, dist_s3 = self.vae_s3.encode(joint_ref_s3, lengths)
                z_s4, dist_s4 = self.vae_s4.encode(joint_ref_s4, lengths)
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            # text encode
            cond_emb = self.text_encoder(text)
        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion process return with noise and noise_pred
        n_set_s4 = self._diffusion_process(z_s4, cond_emb, lengths, scale='s4', coarse_motion=None)
        with torch.no_grad():
            z_s4_rst = self._diffusion_reverse(cond_emb, lengths, 's4', coarse_motion=None)

        n_set_s3 = self._diffusion_process(z_s3, cond_emb, lengths, scale='s3', coarse_motion=z_s4_rst)
        with torch.no_grad():
            z_s3_rst = self._diffusion_reverse(cond_emb, lengths, 's3', coarse_motion=z_s4_rst)

        n_set_s2 = self._diffusion_process(z_s2, cond_emb, lengths, scale='s2', coarse_motion=z_s3_rst)
        with torch.no_grad():
            z_s2_rst = self._diffusion_reverse(cond_emb, lengths, 's2', coarse_motion=z_s3_rst)

        n_set_s1 = self._diffusion_process(z_s1, cond_emb, lengths, scale='s1', coarse_motion=z_s2_rst)

        return {**n_set_s1}, {**n_set_s2}, {**n_set_s3}, {**n_set_s4}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]
        feats_ref = batch["motion"]
        joint_ref_s1 = self.feats2joints(feats_ref)
        joint_ref_s2 = self.init_s1_to_s2(joint_ref_s1)
        joint_ref_s3 = self.init_s1_to_s3(joint_ref_s1)
        joint_ref_s4 = self.init_s1_to_s4(joint_ref_s1)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                z_s1_ref, dist_s1 = self.vae_s1.encode(feats_ref, lengths)
                z_s2_ref, dist_s2 = self.vae_s2.encode(joint_ref_s2, lengths)
                z_s3_ref, dist_s3 = self.vae_s3.encode(joint_ref_s3, lengths)
                z_s4_ref, dist_s4 = self.vae_s4.encode(joint_ref_s4, lengths)

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                        dtype=batch['action'].dtype))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z_s4 = self._diffusion_reverse(cond_emb, lengths, 's4', coarse_motion=None)
            z_s3 = self._diffusion_reverse(cond_emb, lengths, 's3', coarse_motion=z_s4)
            z_s2 = self._diffusion_reverse(cond_emb, lengths, 's2', coarse_motion=z_s3)
            z_s1 = self._diffusion_reverse(cond_emb, lengths, 's1', coarse_motion=z_s2)

        if self.vae_type in ["mld", "vposert", "actor"]:
            feats_rst = self.vae_s1.decode(z_s1, lengths, 's1')
        elif self.vae_type == "no":
            feats_rst = z_s1.permute(1, 0, 2)
        else:
            raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z_s1.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae_s1.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae_s1.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["m_rst"] = feats_rst
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
            rs_set["joints_rst"] = joints_rst
            rs_set["lat_ref_s1"] = z_s1_ref
            rs_set["lat_rst_s1"] = z_s1
            rs_set["lat_ref_s2"] = z_s2_ref
            rs_set["lat_rst_s2"] = z_s2
            rs_set["lat_ref_s3"] = z_s3_ref
            rs_set["lat_rst_s3"] = z_s3
            rs_set["lat_ref_s4"] = z_s4_ref
            rs_set["lat_rst_s4"] = z_s4
        return rs_set

    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            with torch.no_grad():
                z_s4 = self._diffusion_reverse(text_emb, lengths, 's4', coarse_motion=None, reverse_in_test_flage=True)
                z_s3 = self._diffusion_reverse(text_emb, lengths, 's3', coarse_motion=z_s4, reverse_in_test_flage=True)
                z_s2 = self._diffusion_reverse(text_emb, lengths, 's2', coarse_motion=z_s3, reverse_in_test_flage=True)
                z_s1 = self._diffusion_reverse(text_emb, lengths, 's1', coarse_motion=z_s2, reverse_in_test_flage=True)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z_s1, dist_m = self.vae_s1.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z_s1 = torch.randn_like(z_s1)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae_s1.decode(z_s1, lengths, 's1')
            elif self.vae_type == "no":
                feats_rst = z_s1.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set

    def a2m_eval(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        if self.do_classifier_free_guidance:
            cond_emb = torch.cat((torch.zeros_like(actions), actions))

        if self.stage in ['diffusion', 'vae_diffusion']:
            z = self._final_diffusion_reverse(cond_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert","actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("vae_type must be mcross or actor")

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        mask = batch["mask"]
        joints_rst = self.feats2joints(feats_rst, mask)
        joints_ref = self.feats2joints(motions, mask)
        joints_eval_rst = self.feats2joints_eval(feats_rst, mask)
        joints_eval_ref = self.feats2joints_eval(motions, mask)

        rs_set = {
            "m_action": actions,
            "m_ref": motions,
            "m_rst": feats_rst,
            "m_lens": lengths,
            "joints_rst": joints_rst,
            "joints_ref": joints_ref,
            "joints_eval_rst": joints_eval_rst,
            "joints_eval_ref": joints_eval_ref,
        }
        return rs_set

    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        lengths = batch["length"]
        motions = batch["motion"].detach().clone()
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m_s1"]
                loss = self.losses[split].update(rs_set)
            elif self.stage == "diffusion":
                rs_set_s1,rs_set_s2,rs_set_s3, rs_set_s4 = self.train_diffusion_forward(batch)
                loss = self.losses[split].update(rs_set_s1) + self.losses[split].update(rs_set_s2) + self.losses[
                    split].update(rs_set_s3) + + self.losses[split].update(rs_set_s4)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
                loss = self.losses[split].update(rs_set)
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            if self.condition in ['text', 'text_uncond']:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
