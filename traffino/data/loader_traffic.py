import sys
sys.path.append('/home/gpuadmin/dev/Trajectory_Prediction')


from torch.utils.data import DataLoader

from traffino.data.trajectories_traffic import TrajectoryDataset, seq_collate # traffic ligth 적용을 위한 dataset


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
        collate_fn=seq_collate)
    return dset, loader
