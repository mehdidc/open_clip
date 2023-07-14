import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler


# FSDP
major, minor, *rest = torch.__version__.split(".")
if (int(major), int(minor)) >= (2, 1):
    # FSDP is only supported for torch >= 2.1
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
    from torch.distributed.fsdp.api  import StateDictType, FullStateDictConfig, FullOptimStateDictConfig, ShardingStrategy, LocalStateDictConfig, LocalOptimStateDictConfig, ShardedStateDictConfig, ShardedOptimStateDictConfig
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        offload_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)
try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip.model import CLIP
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.checkpoint import PeriodicCheckpointer

#https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/fsdp/__init__.py#L86
class FSDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            data["model"] = self.model.state_dict()

        # data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.{self.rank}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{self.rank}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{self.rank}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{self.rank}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore



LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def main(args):
    args = parse_args(args)
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest and not args.fsdp_only_save_full_checkpoint:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device='cpu' if args.fsdp_init_on_cpu else device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )
    if args.distill:
        # FIXME: currenlty assumes the model your distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    # Prepare parameters to decay
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    parameters_to_decay = set(n for n, p in model.named_parameters() if not exclude(n,p))

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if not args.fsdp:
        if args.lock_image:
            # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
            model.lock_image_tower(
                unlocked_groups=args.lock_image_unlocked_groups,
                freeze_bn_stats=args.lock_image_freeze_bn_stats)
        if args.lock_text:
            model.lock_text_tower(
                unlocked_layers=args.lock_text_unlocked_layers,
                freeze_layer_norm=args.lock_text_freeze_layer_norm)
        if args.lock_text_decoder:
            model.lock_text_decoder_tower(
                unlocked_layers=args.lock_text_decoder_unlocked_layers,
                freeze_layer_norm=args.lock_text_freeze_layer_norm)
        if args.grad_checkpointing:
            model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.fsdp:
            if is_master(args):
                logging.info(f"Before FSDP number of params: {sum(p.numel() for p in model.parameters())}")
                logging.info(f"Before FSDP memory allocated: {torch.cuda.memory_allocated()/1024**3:.4} GB")
            type_name_to_class = {
                "amp": torch.float16,
                "amp_bf16": torch.bfloat16,
                "amp_bfloat16": torch.bfloat16,
                "fp16":  torch.float16,
                "fp32": torch.float32,
            }
            mixed_precision = MixedPrecision(
                reduce_dtype=type_name_to_class[args.fsdp_gradient_reduction_precision],
            )
            layers = set()
            for module in model.modules():
                name = module.__class__.__name__
                for layer in args.fsdp_layers_to_wrap:
                    if re.match(layer, name):
                        layers.add(module.__class__)

            if is_master(args):
                logging.info(f"FSDP Wrapped layers: {layers}")
            strategy = {
                "full": ShardingStrategy.FULL_SHARD,
                "hybrid": ShardingStrategy.HYBRID_SHARD,
            }[args.fsdp_sharding_strategy]
            wrapper_kwargs = dict(
                mixed_precision=mixed_precision,
                limit_all_gathers=args.fsdp_limit_allgathers,
                cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload),
                auto_wrap_policy=ModuleWrapPolicy(layers),
                use_orig_params=True,
                sync_module_states=True,
                device_id=device,
                sharding_strategy=strategy,
            )
            if args.lock_image:
                model.lock_image_tower(
                    unlocked_groups=args.lock_image_unlocked_groups,
                    freeze_bn_stats=args.lock_image_freeze_bn_stats)
            if args.lock_text:
                model.lock_text_tower(
                    unlocked_layers=args.lock_text_unlocked_layers,
                    freeze_layer_norm=args.lock_text_freeze_layer_norm)
            if args.lock_text_decoder:
                model.lock_text_decoder_tower(
                    unlocked_layers=args.lock_text_decoder_unlocked_layers,
                    freeze_layer_norm=args.lock_text_freeze_layer_norm)
            print(model.logit_scale.shape)
            if args.fsdp_sharded_state_dict:
                model.logit_scale = torch.nn.Parameter(torch.Tensor([model.logit_scale.item()]))
            model = FSDP(model, **wrapper_kwargs)
            if is_master(args):
                logging.info(f"After FSDP number of params: {sum(p.numel() for p in model.parameters())}")
                logging.info(f"After FSDP memory allocated: {torch.cuda.memory_allocated()/1024**3:.4} GB")
                
            if args.grad_checkpointing:
                layers_grad_checkpoint = set()
                for module in model.modules():
                    name = module.__class__.__name__
                    for layer in args.fsdp_layers_to_grad_checkpoint:
                        if re.match(layer, name):
                            layers_grad_checkpoint.add(module.__class__)
                non_reentrant_wrapper = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
                if args.fsdp_cpu_offload:
                    wrapper = lambda module:offload_wrapper(non_reentrant_wrapper(module))
                else:
                    wrapper = non_reentrant_wrapper
                check_fn = lambda submodule: (any(isinstance(submodule, layer) for layer in layers_grad_checkpoint))
                apply_activation_checkpointing(
                    model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn
                )
        else:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args['static_graph'] = True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
            if args.distill:
                dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'
        named_parameters = list(model.named_parameters())
        if args.fsdp:
            def _param_name_without_fsdp_prefix(n):
                n = n.replace("_fsdp_wrapped_module.", "")
                n = n.replace("._checkpoint_wrapped_module", "")
                return n
            gain_or_bias_params = [p for n, p in named_parameters if _param_name_without_fsdp_prefix(n) not in parameters_to_decay and p.requires_grad] 
            rest_params = [p for n, p in named_parameters if _param_name_without_fsdp_prefix(n) in parameters_to_decay and p.requires_grad]
        else:
            gain_or_bias_params = [p for n, p in named_parameters if n not in parameters_to_decay and p.requires_grad] 
            rest_params = [p for n, p in named_parameters if n in parameters_to_decay and p.requires_grad]
            
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if args.fsdp:
            scaler = ShardedGradScaler()
        else:
            scaler = GradScaler() if args.precision == "amp" else None
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.fsdp:
        if args.fsdp_sharded_state_dict:

            if args.fsdp_sharded_state_dict_type == "local":
                FSDP.set_state_dict_type(
                    model,
                    StateDictType.LOCAL_STATE_DICT,
                    LocalStateDictConfig(offload_to_cpu=False),
                    LocalOptimStateDictConfig(offload_to_cpu=False), 
                )
            elif args.fsdp_sharded_state_dict_type == "sharded":
                FSDP.set_state_dict_type(
                    model,
                    StateDictType.SHARDED_STATE_DICT,
                    ShardedStateDictConfig(offload_to_cpu=True),
                    ShardedOptimStateDictConfig(offload_to_cpu=True), 
                ) 
            else:
                raise ValueError("Invalid fsdp_sharded_state_dict_type")
        else:
            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=False, offload_to_cpu=True),
            )
    

    
    if args.resume is not None:
        if args.fsdp_sharded_state_dict:
            
            print("RESUME FROM", args.resume)
            """
            
            checkpoint = {
                'state_dict': model.state_dict(),
                'epoch': 0,
            }
            dist_cp.load_state_dict(state_dict=checkpoint, storage_reader=dist_cp.FileSystemReader(args.resume))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=checkpoint["state_dict"],
                optimizer_key="optimizer",
                storage_reader=dist_cp.FileSystemReader(args.resume),
            )
            flattened_state_dict = FSDP.optim_state_dict_to_load(model, optimizer, optim_state["optimizer"])
            optimizer.load_state_dict(flattened_state_dict)
            """
            sd = torch.load(os.path.join(args.resume, f"{args.rank}.pt"), map_location='cpu')

            print(sd['state_dict'].keys())
            model.load_state_dict(sd['state_dict'], strict=False)

            optimizer.load_state_dict(sd['optimizer'])
            if scaler is not None and 'scaler' in sd:
                scaler.load_state_dict(sd['scaler'])
            start_epoch = sd['epoch']
        else:
            checkpoint = pt_load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    if args.fsdp:
                        optimizer_state_dict = checkpoint["optimizer"]
                        optimizer_state_dict['state']['logit_scale']['exp_avg'] = optimizer_state_dict['state']['logit_scale']['exp_avg'].view(1)
                        optimizer_state_dict['state']['logit_scale']['exp_avg_sq'] = optimizer_state_dict['state']['logit_scale']['exp_avg_sq'].view(1)
                        sharded_state_dict = FSDP.optim_state_dict_to_load(model, optimizer, optimizer_state_dict)
                        optimizer.load_state_dict(sharded_state_dict)
                    else:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    if args.fsdp_only_save_full_checkpoint:
        """
        from clip_benchmark.datasets.builder import build_dataset
        from clip_benchmark.metrics import zeroshot_classification
        ds = build_dataset(
            dataset_name="wds/imagenet1k", 
            root="/p/fastdata/mmlaion/vtab_plus_wds/wds_imagenet1k", 
            transform=preprocess_val, 
        )
        dl = torch.utils.data.DataLoader(
            ds.batched(64), 
            batch_size=None, 
            shuffle=False, num_workers=4,
        )
        zeroshot_templates = ds.templates
        classnames = ds.classes
        with torch.no_grad():
            metrics = zeroshot_classification.evaluate(
                model, 
                dl, 
                get_tokenizer(args.model), 
                classnames, zeroshot_templates, 
                device="cuda", 
                amp=True,
            )
        print(metrics)
        """

        FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=False, offload_to_cpu=True),
        )
        sd = model.state_dict()
        checkpoint = {
            "state_dict": sd,
            "name": args.name
        }
        if args.rank == 0:   
            torch.save(checkpoint, os.path.join(args.resume, f"full.pt"))
        torch.distributed.barrier()
        sys.exit(0)
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'


    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)
    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and (is_master(args) or args.fsdp)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    loss = create_loss(args)

    """   
    if args.fsdp_sharded_state_dict:
        checkpointer = FSDPCheckpointer(model, args.checkpoint_path, optimizer=optimizer, save_to_disk=True)
        checkpointer.rank = args.rank
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=1,
            max_iter=args.epochs,
            max_to_keep=3,
        )
        start_epoch = checkpointer.resume_or_load(args.checkpoint_path, resume=True).get("iteration", 0)
     """

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        #periodic_checkpointer.step(completed_epoch)
        
        
        if args.save_logs:
  
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": FSDP.optim_state_dict(model, optimizer) if (args.fsdp and not args.fsdp_sharded_state_dict) else optimizer.state_dict(),
                "optimizer": optimizer.state_dict(),

            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if args.fsdp_sharded_state_dict:
                folder = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt")
                os.makedirs(folder, exist_ok=True)
                torch.save(checkpoint_dict, os.path.join(folder, f"{args.rank}.pt"))
                if args.save_most_recent:
                    folder = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                    os.makedirs(folder, exist_ok=True)
                    torch.save(checkpoint_dict, os.path.join(folder, f"{args.rank}.pt"))
                """
                
                dist_cp.save_state_dict(
                    state_dict=checkpoint_dict,
                    storage_writer=dist_cp.FileSystemWriter(folder),
                )
                if args.save_most_recent:
                    folder = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                    os.makedirs(folder, exist_ok=True)
                    dist_cp.save_state_dict(
                        state_dict=checkpoint_dict,
                        storage_writer=dist_cp.FileSystemWriter(folder),
                    )
                """
            else:
                if is_master(args):
                    if completed_epoch == args.epochs or (
                        args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                    ):
                        torch.save(
                            checkpoint_dict,
                            os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                        )
                    if args.delete_previous_checkpoint:
                        previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                        if os.path.exists(previous_checkpoint):
                            os.remove(previous_checkpoint)

                    if args.save_most_recent:
                        # try not to corrupt the latest checkpoint if save fails
                        tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                        latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                        torch.save(checkpoint_dict, tmp_save_path)
                        os.replace(tmp_save_path, latest_save_path)
        
    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path) and not args.fsdp_only_save_full_checkpoint:
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
