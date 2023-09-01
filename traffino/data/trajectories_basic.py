import logging
import os
import math

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        obs_seq_list, 
        pred_seq_list, 
        obs_seq_rel_list, 
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list

    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)

    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)

    
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)

    
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)

    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    # img_list = torch.cat(img_list, dim=0)
    
    out = [
        obs_traj, 

        pred_traj, 

        obs_traj_rel, 

        pred_traj_rel, 

        
        non_linear_ped,
        
        loss_mask, 
        seq_start_end,

    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    print(_path)
    delim = '\t'
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, 
             threshold
             ):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len) # np.linspace(시작점, 끝점, 구간 내 숫자 개수)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold: # error
       return 1.0
    else:
       return 0.0
    return 0.0

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, 
        data_dir,
        obs_len=8, 
        pred_len=12, 
        skip=1, 
        threshold=0.002, 
        min_agent=1, 
        delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format ###  파일명을 제외한 디렉토리 경로를 주어야 함
        <frame_id> <agent_id> <x> <y> <speed> <tan_acc> <lat_acc> <angle> <tl_code> <time>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_agent: Minimum number of agents that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir 
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len # 20
        self.delim = delim
        
        # temp, _= os.path.split(self.data_dir)
        # img_dir = temp+ '_img' # path = '/home/gpuadmin/dev/traj_pred/Trajectory_Prediction/traffino/datasets/waterloo/train_img'
        
        

        all_files = os.listdir(self.data_dir) # 디렉토리에 있는 모든 파일을 리스트로 가져옴, path = '/home/gpuadmin/dev/traj_pred/Trajectory_Prediction/traffino/datasets/waterloo/train'
        print(all_files) # data_dir에서 파일을 잘 불러오는 지 확인
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] # data_dir\_path
        all_files = sorted(all_files)
        num_agents_in_seq = []
        
        seq_list = []

        
        seq_list_rel = []
        # seq_list_rel2 = [] # state
        # seq_list_rel3 = [] # traffic 
        
        loss_mask_list = []
        non_linear_agent = []
        folder_num= 770
        for path in all_files:
            
            dir_name, file_name_ext = os.path.split(path) # dir_name='/home/gpuadmin/dev/traj_pred/Trajectory_Prediction/traffino/datasets/waterloo/train/'
            file_name, ext = os.path.splitext(file_name_ext)
            # path = dir_name + '\\str(folder_num)\\' + file_name + ext 
            # path2 = dir_name + '2/'+ file_name + '2'+ ext
            # path3 = dir_name + '2/'+ file_name + '3'+ ext
            
            data = read_file(path, delim) # 0769_prep.txt (원본, 10 column) # <frame_id> <agent_id> <x> <y> <speed> <tan_acc> <lat_acc> <angle> <tl_code> <time>
            # data2 = read_file(path2, delim) # <frame_id> <agent_id> <speed> <tan_acc> <lat_acc> <angle> 
            # data3 = read_file(path3, delim) # <frame_id> <tl_code> 

            folder_num +=1
            
            frames = np.unique(data[:, 0]).tolist()# np.unique : 중복된 값을 제거한 배열을 반환함, # 모든 행 0번째 열(<frame_id>)만 slicing
            # print(f"len(frames) : {len(frames)}")
            # print(f"frame : {frames}")
            # frame_data_all = []
            frame_data = []

            
            for frame in frames:

                frame_data.append(data[frame == data[:, 0], :4]) # frame_data, 각 frame별 frame_data (agent 정보, x, y) --> 2차원 list

                # Get image_tensor
                dir_num = 769
                # img = Image.open(os.path.join(img_dir, str(dir_num), str(dir_num) + "_frame_" + str(int(frame-1))+ ".jpg" ))
                to_tensor = transforms.ToTensor()

                

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # sequences의 숫자 // 가까운 수의 상위 정수로 반올림 (7994-20)/1 = 7975

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(                         # num_sequnce만큼 끊어서 curr seq_data 생성 // frame_data[0:20], frame_data[1:21], frame_data[2:22]
                    frame_data[idx : idx + self.seq_len], axis=0
                )

            
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])       # 현재 seq에 있는 agents 목록, 모든 행의 1번째(agent 정보) 열 slicing

                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 2,        # (현재 seq에 있는 agents 개수, 2, seq_len) 
                                         self.seq_len))

                
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # (현재 seq에 있는 agents 개수, 2, seq_len) 
                                       
                curr_loss_mask = np.zeros((len(agents_in_curr_seq),
                                           self.seq_len))
                
                num_agents_considered = 0
                _non_linear_agent = []
                
                for _, agent_id in enumerate(agents_in_curr_seq):           # 현재 seq에 있는 agents 목록
                    curr_agent_seq = curr_seq_data[ curr_seq_data[:, 1] ==  # 현재 seq에 있는 agent의 위치 seqeunce curr_agent_seq: (20, 4)
                                                 agent_id, :]

                    
                    curr_agent_seq = np.around(curr_agent_seq, decimals=4)  # (20, 4)

                    agent_front = frames.index(curr_agent_seq[0, 0]) - idx
                    agent_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    
                    if agent_end - agent_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    
                    curr_agent_seq = curr_agent_seq
                    
                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                    _idx = num_agents_considered
                    
                    curr_seq[_idx, :, agent_front:agent_end] = curr_agent_seq

                    curr_seq_rel[_idx, :, agent_front:agent_end] = rel_curr_agent_seq
                    
                    # Linear vs Non-Linear Trajectory
                    _non_linear_agent.append(
                        poly_fit(curr_agent_seq, pred_len 
                                 ,threshold
                                 )
                                 )
                    curr_loss_mask[_idx, agent_front:agent_end] = 1
                    num_agents_considered += 1

                if num_agents_considered > min_agent:
                    non_linear_agent += _non_linear_agent
                    num_agents_in_seq.append(num_agents_considered)
                    loss_mask_list.append(curr_loss_mask[:num_agents_considered])
                    
                    seq_list.append(curr_seq[:num_agents_considered])
                    
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])


        self.num_seq = len(seq_list)
        
        seq_list = np.concatenate(seq_list, axis=0)

        
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        # img_list = np.concatenate(img_list, axis=0)
        non_linear_agent = np.asarray(non_linear_agent)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)

        
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)

    
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)

        
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)

        
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_agent = torch.from_numpy(non_linear_agent).type(torch.float)
        # self.img_list = torch.from_numpy(img_list).type(torch.float)
        
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        


    def __len__(self):              # len(train_dset)
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],                # 0
            self.pred_traj[start:end, :],               # 1
            self.obs_traj_rel[start:end, :],            # 2
            self.pred_traj_rel[start:end, :],           # 3
            self.non_linear_agent[start:end],           # 4
            self.loss_mask[start:end, :]                # 5
        ]
        return out