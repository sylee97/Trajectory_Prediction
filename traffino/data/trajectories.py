import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        obs_seq_list, 
        pred_seq_list, 
        obs_seq_rel_list, 
        pred_seq_rel_list,
        non_linear_ped_list, 
        loss_mask_list,
        # cnn_list
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
    # cnn_list = torch.cat(cnn_list, dim=0).repeat(obs_traj.size(1), 1, 1) # 
    out = [
        obs_traj, 
        pred_traj, 
        obs_traj_rel, 
        pred_traj_rel, 
        non_linear_ped,
        loss_mask, 
        seq_start_end,
        # cnn_list
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
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

        all_files = os.listdir(self.data_dir) # 디렉토리에 있는 모든 파일을 리스트로 가져옴, path = 'C:\\Users\\NGN\\dev\\Traffino\\TRAFFINO\\traffino\\datasets'
        print(all_files) # data_dir에서 파일을 잘 불러오는 지 확인
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] # data_dir\_path
        num_agents_in_seq = []
        
        seq_list = []
        seq_list2 = [] # state
        seq_list3 = [] # traffic 
        
        seq_list_rel = []
        seq_list_rel2 = [] # state
        seq_list_rel3 = [] # traffic 
        
        loss_mask_list = []
        non_linear_agent = []
        
        for path in all_files:
            _, file_name_ext = os.path.split(path)
            file_name, ext = os.path.splitext(file_name_ext)
            path2 = file_name + '2'+ ext
            path3 = file_name + '3'+ ext
            
            data = read_file(path, delim) # 0769_prep.txt (원본, 10 column) # <frame_id> <agent_id> <x> <y> <speed> <tan_acc> <lat_acc> <angle> <tl_code> <time>
            data2 = read_file(path2, delim) # <frame_id> <agent_id> <speed> <tan_acc> <lat_acc> <angle> 
            data3 = read_file(path3, delim) # <frame_id> <tl_code> 
            
            frames = np.unique(data[:, 0]).tolist()# np.unique : 중복된 값을 제거한 배열을 반환함, # 모든 행 0번째 열(<frame_id>)만 slicing
            # print(f"len(frames) : {len(frames)}")
            # print(f"frame : {frames}")
            frame_data_all = []
            frame_data = []
            state_data = []
            traffic_data = []
            for frame in frames:
                frame_data_all.append(data[frame == data[:, 0], :4]) # all
                frame_data.append(data[frame == data[:, 0], :4]) # frame_data, 각 frame별 frame_data (agent 정보, x, y) --> 2차원 list
                state_data.append(data2[frame == data2[:, 0], :])
                traffic_data.append(data3[frame == data3[:, 0], ])
                
            # print(f"len(frame_data) : {len(frame_data)}")
            # print(f"frame_data : {frame_data}")
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # sequences의 숫자 // 가까운 수의 상위 정수로 반올림 (7994-20)/1 = 7975

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(                         # num_sequnce만큼 끊어서 curr seq_data 생성 // frame_data[0:20], frame_data[1:21], frame_data[2:22]
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                curr_seq_data2 = np.concatenate(                         # num_sequnce만큼 끊어서 curr seq_data 생성
                    state_data[idx:idx + self.seq_len], axis=0               # ex) seq 1: 1-20 frame, seq 2: 2-21 frame ... 
                )
                curr_seq_data3 = np.concatenate(                         # num_sequnce만큼 끊어서 curr seq_data 생성
                    traffic_data[idx:idx + self.seq_len], axis=0               # ex) seq 1: 1-20 frame, seq 2: 2-21 frame ... 
                )
            
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])       # 현재 seq에 있는 agents 목록, 모든 행의 1번째(agent 정보) 열 slicing

                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 2,        # (현재 seq에 있는 agents 개수, 2, seq_len) 
                                         self.seq_len))
                curr_seq_rel2 = np.zeros((len(agents_in_curr_seq), 4,        # (현재 seq에 있는 agents 개수, 4, seq_len) --> state 저장용
                                         self.seq_len))
                curr_seq_rel3 = np.zeros((len(agents_in_curr_seq), 1,        # (현재 seq에 있는 agents 개수, 1, seq_len) --> traffic light 저장용
                                         self.seq_len))
                
                curr_seq = np.zeros((len(agents_in_curr_seq), 2, self.seq_len)) # (현재 seq에 있는 agents 개수, 2, seq_len) 
                curr_seq2 = np.zeros((len(agents_in_curr_seq), 4, self.seq_len)) # (현재 seq에 있는 agents 개수, 4, seq_len) --> state 저장용
                curr_seq3 = np.zeros((len(agents_in_curr_seq), 1, self.seq_len)) # (현재 seq에 있는 agents 개수, 1, seq_len) --> traffic light 저장용
                                                
                curr_loss_mask = np.zeros((len(agents_in_curr_seq),
                                           self.seq_len))
                
                num_agents_considered = 0
                _non_linear_agent = []
                
                for _, agent_id in enumerate(agents_in_curr_seq):           # 현재 seq에 있는 agents 목록
                    curr_agent_seq = curr_seq_data[ curr_seq_data[:, 1] ==  # 현재 seq에 있는 agent의 위치 seqeunce curr_agent_seq: (20, 4)
                                                 agent_id, :]
                    curr_agent_seq2 = curr_seq_data2[ curr_seq_data2[:, 1] ==  # 현재 seq에 있는 agent의 위치 seqeunce curr_agent_seq: (20, 4)
                                                 agent_id, :]
                    curr_agent_seq3 = curr_seq_data3[ curr_seq_data3[:, 1] ==  # 현재 seq에 있는 agent의 위치 seqeunce curr_agent_seq: (20, 4)
                                                 agent_id, :]
                    
                    curr_agent_seq = np.around(curr_agent_seq, decimals=4)  # (20, 4)
                    curr_agent_seq2 = np.around(curr_agent_seq2, decimals=4)  # (20, 4)
                    curr_agent_seq3 = np.around(curr_agent_seq3, decimals=4)  # (20, 4)
                    
                    agent_front = frames.index(curr_agent_seq[0, 0]) - idx
                    agent_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    
                    if agent_end - agent_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    curr_agent_seq2 = np.transpose(curr_agent_seq2[:, 2:])
                    curr_agent_seq3 = np.transpose(curr_agent_seq3[:, 2:])
                    
                    curr_agent_seq = curr_agent_seq
                    
                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                    _idx = num_agents_considered
                    
                    curr_seq[_idx, :, agent_front:agent_end] = curr_agent_seq
                    curr_seq2[_idx, :, agent_front:agent_end] = curr_agent_seq2
                    curr_seq3[_idx, :, agent_front:agent_end] = curr_agent_seq3
                    
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
                    seq_list2.append(curr_seq2[:num_agents_considered])
                    seq_list3.append(curr_seq3[:num_agents_considered])
                    
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    seq_list_rel2.append(curr_seq_rel2[:num_agents_considered])
                    seq_list_rel3.append(curr_seq_rel3[:num_agents_considered])

        self.num_seq = len(seq_list)
        
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list2 = np.concatenate(seq_list2, axis=0)
        seq_list3 = np.concatenate(seq_list3, axis=0)
        
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_rel2 = np.concatenate(seq_list_rel2, axis=0)
        seq_list_rel3 = np.concatenate(seq_list_rel3, axis=0)
        
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_agent = np.asarray(non_linear_agent)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_state = torch.from_numpy(
            seq_list2[:, :, :self.obs_len]).type(torch.float)  
        self.obs_traffic = torch.from_numpy(
            seq_list3[:, :, :self.obs_len]).type(torch.float)    
        
        self.pred_traj = torch.from_numpy(
            seq_list2[:, :, self.obs_len:]).type(torch.float)
        self.pred_state = torch.from_numpy(
            seq_list2[:, :, self.obs_len:]).type(torch.float)
        self.pred_traffic = torch.from_numpy(
            seq_list3[:, :, self.obs_len:]).type(torch.float)
    
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.obs_state_rel = torch.from_numpy(
            seq_list_rel2[:, :, :self.obs_len]).type(torch.float)
        self.obs_traffic_rel = torch.from_numpy(
            seq_list_rel3[:, :, :self.obs_len]).type(torch.float)
        
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.pred_state_rel = torch.from_numpy(
            seq_list_rel2[:, :, self.obs_len:]).type(torch.float)
        self.pred_traffic_rel = torch.from_numpy(
            seq_list_rel3[:, :, self.obs_len:]).type(torch.float)
        
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_agent = torch.from_numpy(non_linear_agent).type(torch.float)
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
            self.obs_state[start:end, :],               # 1
            self.obs_traffic[start:end, :],             # 2
                        
            self.pred_traj[start:end, :],               # 3
            self.pred_state[start:end, :],              # 4
            self.pred_traffic[start:end, :],            # 5
            
            self.obs_traj_rel[start:end, :],            # 6
            self.obs_state_rel[start:end, :],           # 7
            self.obs_traffic_rel[start:end, :],         # 8
                        
            self.pred_traj_rel[start:end, :],           # 9
            self.pred_state_rel[start:end, :],          # 10         
            self.pred_traffic_rel[start:end, :],        # 11     
                        
            self.non_linear_agent[start:end],           # 12
            self.loss_mask[start:end, :]                # 13
        ]
        return out