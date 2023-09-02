import torch, gc
import torch.nn as nn
# import torchvision.models as models
from torchvision.models import (
#                                 mobilenet_v2, # --> not in torchvision v0.2.1
                                  resnet18, resnet34, resnet50, resnet101, resnet152
#                                 resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, 
#                                 wide_resnet50_2, wide_resnet101_2,
#                                 densenet161
                                )
from typing import Tuple

from torch.nn import functional as f



def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim.
    :param index: Where to trim the network. Counted from the last layer.
    """
    assert index < 0, f"Param index must be negative. Received {index}."
    return nn.Sequential(*list(network.children())[:index])

def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]) -> int:
    """ Helper to calculate the shape of the fully-connected regression layer. """
    tensor = torch.ones(1, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]

RESNET_VERSION_TO_MODEL = {'resnet18': resnet18, 'resnet34': resnet34,
                          'resnet50': resnet50, 'resnet101': resnet101,
                          'resnet152': resnet152}


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)
        
class ResNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: resnet18, resnet34, resnet50, resnet101, resnet152.
    """


    def __init__(self, version: str):
        """
        Inits ResNetBackbone
        :param version: resnet version to use.
        """
        super().__init__()


        if version not in RESNET_VERSION_TO_MODEL:
            raise ValueError(f'Parameter version must be one of {list(RESNET_VERSION_TO_MODEL.keys())}'
                             f'. Received {version}.')


        self.backbone = trim_network_at_index(RESNET_VERSION_TO_MODEL[version](), -1)


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet50,
            the shape is [batch_size, 2048].
        """
        backbone_features = self.backbone(input_tensor).cuda()
        
        return torch.flatten(backbone_features, start_dim=1)
    
STATE_DIM = 4

default_backbone = ResNetBackbone('resnet18')

class SceneFusionTraffic(nn.Module):
    def __init__(self, backbone: nn.Module = default_backbone,
                 n_hidden_layers: int = 1024,
                 input_shape: Tuple[int, int, int] = (3, 1280, 720)):
        super(SceneFusionTraffic, self).__init__()
        
        self.backbone = backbone  
        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        
        self.fc1 = nn.Linear(backbone_feature_dim + 1, n_hidden_layers)
               
    def forward(self, image_tensor: torch.Tensor,
                # agent_state_vector: torch.Tensor, 
                traffic_light: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param image_tensor: Tensor of images shape [batch_size, n_channels, length, width].
        :param agent_state_vector: Tensor of floats representing the agent state with traffic light.
            [batch_size, 5].
        :return: Tensor of dimension [batch_size, number_of_modes * number_of_predictions_per_mode + number_of_modes]
            storing the predicted trajectory and mode probabilities. Mode probabilities are normalized to sum
            to 1 during inference.
        """

        backbone_features = self.backbone(image_tensor)
        fusion_features = torch.cat([backbone_features, traffic_light], dim=1).cuda()


        x = f.relu(self.fc1(fusion_features)) ###
        return x

class TrajEncoder(nn.Module):
    """ """
    def __init__(
        self, h_dim=64, embedding_dim=64
        , mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(TrajEncoder, self).__init__() # Module 클래스 속성을 초기화 함
    
        self.mlp_dim=1024    
        self.h_dim = h_dim # 64
        self.embedding_dim = embedding_dim # embedding to lstm cell : e_{i}^t --> 64
        self.num_layers = num_layers # single layer to get a fixed length vector e_{i}^t --> 1
        
        self.encoder = nn.LSTM( # input_size, hidden_size, num_layers
            embedding_dim, h_dim, num_layers # num_layers=1
        )
        
        self.spatial_embedding = nn.Linear( # location (x_{i}^t, y_{i}^t) + tl_code_{i}^t --> embeddings
            2, embedding_dim # input sample size, output sample size
        ) 
    
    def init_hidden(self, batch):
        
        h = torch.zeros(1, batch, self.h_dim).cuda()
        c = torch.zeros(1, batch, self.h_dim).cuda()
        return (h, c)
    
    def forward(self, obs_traj):
        """ 
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        
        # Encode observed Trajectory
        batch = obs_traj.size(1) # npeds
        # total = batch * (MAX_PEDS if padded else 1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1,2))
        obs_traj_embedding = obs_traj_embedding.contiguous().view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch) # self.init_hidden(total)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h 

class StateEncoder(nn.Module):
    """ """
    def __init__(
        self, h_dim=64, embedding_dim=64
        , mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(StateEncoder, self).__init__() # Module 클래스 속성을 초기화 함
    
        self.mlp_dim=1024    
        self.h_dim = h_dim # 64
        self.embedding_dim = embedding_dim # embedding to lstm cell : e_{i}^t --> 64
        self.num_layers = num_layers # single layer to get a fixed length vector e_{i}^t --> 1
        
        self.encoder = nn.LSTM( # input_size, hidden_size, num_layers
            embedding_dim, h_dim, num_layers # num_layers=1
        )
        
        self.spatial_embedding = nn.Linear( # location (x_{i}^t, y_{i}^t) + tl_code_{i}^t --> embeddings
            4, embedding_dim # input sample size, output sample size
        ) 
    
    def init_hidden(self, batch):
        
        h = torch.zeros(1, batch, self.h_dim).cuda()
        c = torch.zeros(1, batch, self.h_dim).cuda()
        return (h, c)
    
    def forward(self, obs_state):
        """ 
        Inputs:
        - obs_state: Tensor of shape (obs_len, batch, xxx)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        
        # Encode observed Trajectory
        batch = obs_state.size(1) # npeds
        # total = batch * (MAX_PEDS if padded else 1)
        obs_state_embedding = self.spatial_embedding(obs_state.contiguous().view(-1,4))
        obs_state_embedding = obs_state_embedding.contiguous().view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch) # self.init_hidden(total)
        output, state = self.encoder(obs_state_embedding, state_tuple)
        final_h2 = state[0]
        return final_h2 
    
class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in Social GAN paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim # 64
        self.bottleneck_dim = bottleneck_dim # 1024
        self.embedding_dim = embedding_dim # 64

        mlp_pre_dim = embedding_dim + h_dim # 128
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim] # [128, 512, 1024]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
            )

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 3D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.contiguous().view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.contiguous().view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.contiguous().view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h
    
class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, num_layers=1, mlp_dim=1024,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()
        
        self.seq_len = seq_len 
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim 
        self.pool_every_timestep = pool_every_timestep
        self.mlp_dim = mlp_dim 
        
        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers # ,dropout=dropout # 1 -> num_layers
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim = self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 3)
        - last_pos_rel : Tensor of shape (batch, 3)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 3)
        """
        batch = last_pos.size(0) # npred
        pred_traj_fake_rel=[]
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.contiguous().view(1, batch, self.embedding_dim)
         
            
        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.contiguous().view(-1, self.h_dim))
            curr_pos = rel_pos+last_pos
                
            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.contiguous().view(-1, self.h_dim), pool_h], dim=1
                )
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple=(decoder_h, state_tuple[1])
            
            embedding_input=rel_pos
            
            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.contiguous().view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.contiguous().view(batch, -1))
            last_pos=curr_pos 
            
        pred_traj_fake_rel=torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]

# class ContextAttention(nn.Module):
#         def __init__(self):
#             super(ContextFusionPool, self).__init__()
#             self.h_dim = 64
#             self.bottleneck_dim = 1024
#             self.embedding_dim = 64
        
#             mlp_pre_dim = self.embedding_dim + self.h_dim # 64+64
#             mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim] # [80,512,32]
#             self.spatial_embedding = nn.Linear(2, self.embedding_dim)
#             self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)
#             self.attn = nn.Linear(MAX_PEDS*self.bottleneck_dim, MAX_PEDS) # 64*32, 64


class TrajectoryGenerator(nn.Module): 
    def __init__(                           # self 제외 18개
        self, 
        obs_len, 
        pred_len, 
        embedding_dim=64, 
        encoder_h_dim=64, 
        decoder_h_dim=128, 
        mlp_dim=1024, 
        num_layers=1,
        noise_dim=(0, ),
        bottleneck_dim=1024,
        noise_type='gaussian', 
        noise_mix_type='ped', 
        pooling_type= 'pool_net' ,# None,
        pool_every_timestep=True, 
        dropout=0.0, 
        activation='relu', 
        batch_norm=True, 
        neighborhood_size=2.0, 
        grid_size=8,
    ):
        super(TrajectoryGenerator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        
        self.encoder = TrajEncoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim, # 64
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.encoder2 = StateEncoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim, # 64
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        # self.encoder3 = TrafficEncoder(
        #     embedding_dim=embedding_dim,
        #     h_dim=encoder_h_dim, # 64
        #     mlp_dim=mlp_dim,
        #     num_layers=num_layers,
        #     dropout=dropout
        # )

        # self.sft = SceneFusionTraffic(
        #     backbone= default_backbone,
        #     n_hidden_layers = 1024,
        #     input_shape = (3, 1280, 720)
        # )
        
        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim, # 128
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim, # 1024
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            # grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )
        
        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim, # 64
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim, # 1024
                activation=activation,
                batch_norm=batch_norm
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            # input_dim = encoder_h_dim + bottleneck_dim
            input_dim = 2* encoder_h_dim + bottleneck_dim # 64 + 1024 = 1088 ->64*2+1024=1152
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].contiguous().view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h    
    
    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False
        
    
    def forward(self, obs_traj, obs_traj_rel, seq_start_end, 
                obs_state, 
                user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        # - scene_state_tfl: Scene feautre with state, traffic light 
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        
        final_encoder_h2 = self.encoder2(obs_state) # obs_state_rel이 아닌 obs_state 입력
        # final_encoder_h3 = self.encoder3(obs_traj_rel)
        
        
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos) # batch, bottlenet=1024
            # sft = self.sft(image_tensor, traffic_light)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.contiguous().view(-1, self.encoder_h_dim), # 64, location
                 final_encoder_h2.contiguous().view(-1, self.encoder_h_dim), # 64, state
                 # final_encoder_h3.view(-1, self.encoder_h_dim), 
                 pool_h 
                # ,sft
                 ]
                 , dim=1) # 합치는 부분
            
        # else:
        #     mlp_decoder_context_input = final_encoder_h.view(
        #         -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel
    
class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = TrajEncoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim, # 1024
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 3)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 3)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores


