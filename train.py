import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import tqdm

import modelsrc.commons as commons
import utils
from dataloader import AudioDataLoader, DataCollate
from modelsrc.models3 import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    AVAILABLE_FLOW_TYPES,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel import mel_spectrogram_torch, spec_to_mel_torch
import gc


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6060"

    hps = utils.get_hparams()
    run(0, 1, hps)


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    train_dataset = AudioDataLoader(f'.\\train_wavs', hparams=hps.data)
    collate_fn = DataCollate()
    train_loader = DataLoader(train_dataset, num_workers=4, shuffle=False, pin_memory=True, batch_size=hps.train.batch_size, collate_fn=collate_fn)
    if rank == 0:
        eval_dataset = AudioDataLoader(f'.\\val_wavs', hparams=hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=4, shuffle=False, pin_memory=False, batch_size=hps.train.batch_size, collate_fn=collate_fn)
    # some of these flags are not being used in the code and directly set in hps json file.
    # they are kept here for reference and prototyping.

    if (
        "use_transformer_flows" in hps.model.keys()
        and hps.model.use_transformer_flows == True
    ):
        use_transformer_flows = True
        transformer_flow_type = hps.model.transformer_flow_type
        print(f"Using transformer flows {transformer_flow_type} for VITS2")
        assert (
            transformer_flow_type in AVAILABLE_FLOW_TYPES
        ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
    else:
        print("Using normal flows for VITS1")
        use_transformer_flows = False

    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder == True
    ):
        if hps.data.n_speakers == 0:
            print("Warning: use_spk_conditioned_encoder is True but n_speakers is 0")
        print(
            "Setting use_spk_conditioned_encoder to False as model is a single speaker model"
        )
        use_spk_conditioned_encoder = False
    else:
        print("Using normal encoder for VITS1")
        use_spk_conditioned_encoder = False

    net_g = SynthesizerTrn(
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    try:
        net_g, optim_g, hps.train.learning_rate, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        net_d, optim_d, hps.train.learning_rate, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d = nets
    optim_g, optim_d  = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    net_g.train()
    net_d.train()

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc="Loading train data")
    else:
        loader = train_loader
    for batch_idx, (vec, spec, cqt, lengths, wav_norm, wav_length, sid) in enumerate(
        loader
    ):
        vec = vec.cuda(rank, non_blocking=True)
        spec = spec.cuda(rank, non_blocking=True)
        cqt = cqt.cuda(rank, non_blocking=True)
        wav = wav_norm.cuda(rank, non_blocking=True)
        speaker = sid.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                audio,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(vec, spec, cqt, lengths, sid=speaker)


            y_mel = commons.slice_segments(
                spec, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                audio.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                wav, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, audio.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, audio)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )

                # if net_dur_disc is not None:
                #   scalar_dict.update({"loss/dur_disc_r" : f"{losses_dur_disc_r}"})
                #   scalar_dict.update({"loss/dur_disc_g" : f"{losses_dur_disc_g}"})
                #   scalar_dict.update({"loss/dur_gen" : f"{loss_dur_gen}"})

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        spec[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                utils.remove_old_checkpoints(hps.model_dir, prefixes=["G_*.pth", "D_*.pth"])
        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (vec, spec, cqt, lengths, wav_norm, wav_length, sid) in enumerate(
            eval_loader
        ):
            rank = 0
            wav = wav_norm.cuda(rank, non_blocking=True)
            vec = vec.cuda(rank, non_blocking=True)
            cqt = cqt.cuda(rank, non_blocking=True)
            speaker = sid.cuda(rank, non_blocking=True)
            lengths = lengths.cuda(rank, non_blocking=True)

            audio = generator.infer(vec, cqt, lengths, sid=speaker)

            mel = spec

            y_hat_mel = mel_spectrogram_torch(
                audio.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
        image_dict = {
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        }
        audio_dict = {"gen/audio": audio[0][0]}
        if global_step == 0:
            image_dict.update(
                {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
            )
            audio_dict.update({"gt/audio": wav[0][0]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    gc.collect()
    torch.cuda.empty_cache()
    generator.train()


if __name__ == "__main__":
    main()