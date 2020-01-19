import sys
sys.path.append('../')
import hi_vae_functions as hvf

class model_hi_vae():
    def __init__(
        self,
        x_dim,
        hidden_x_dim,
        z_dim,
        s_dim,
        column_types):
        self.graph = {}
        self.column_types = column_types
        self.x_dim = x_dim
        self.hidden_x_dim = hidden_x_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.endecoder = hvf.get_hi_vae_encoder(
            self.graph,
            self.column_types,
            self.x_dim,
            self.hidden_x_dim,
            self.z_dim,
            self.s_dim
        )

    def get_trainable_variables(self):
        return self.endecoder.trainable_variables

