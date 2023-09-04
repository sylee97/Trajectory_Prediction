import sys
# sys.path.append('/home/gpuadmin/dev/Trajectory_Prediction')
sys.path.append('C:/Users/NGN/dev/Traffino/TRAFFINO')

from torch.utils.data import DataLoader

from traffino.data.image_data import ImageDataset, seq_collate
import torchvision.transforms as T

def data_loader(args, path):
    transforms = T.Compose([T.ToTensor()])
    dset = ImageDataset(
        path,
        transforms)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        # shuffle=True,
        num_workers=args.loader_num_workers,
    collate_fn=seq_collate) # variable-length input을 batch로 잘 묶어서 dataloader로 넘겨줌
    return dset, loader
