from .FrameStack import FrameStack_Actor, FrameStack_Critic
from .ConvLSTM import ConvLSTM_Actor, ConvLSTM_Critic
from .AttentionModel import Att_Actor, Att_Critic


ACTOR_REGISTRY = {
    "fStack": FrameStack_Actor,
    "conv-lstm": ConvLSTM_Actor,
    "att": Att_Actor
}

CRITIC_REGISTRY = {
    "fStack": FrameStack_Critic,
    "conv-lstm": ConvLSTM_Critic,
    "att": Att_Critic
}