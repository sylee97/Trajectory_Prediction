import sys
sys.path.append('/home/gpuadmin/dev/Trajectory_Prediction')

from torch.utils.data import DataLoader

from traffino.data.trajectories_basic import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        # shuffle=True,
        num_workers=args.loader_num_workers,
    collate_fn=seq_collate) # variable-length input을 batch로 잘 묶어서 dataloader로 넘겨줌
    return dset, loader
