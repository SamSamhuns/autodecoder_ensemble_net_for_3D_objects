# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class AutoDecoder(nn.Module):
    """
    AutoDecoder NN to learn point drift (latent encoding) between two 3D shapes
    """

    def __init__(self,  encoding_size=256, point_dim=3):
        super(AutoDecoder, self).__init__()
        self.fc1 = nn.Conv1d(encoding_size + point_dim, 128, 1)
        self.fc2 = nn.Conv1d(128, 64, 1)
        self.fc3 = nn.Conv1d(64, point_dim, 1)

    def forward(self, X, encoding):
        num_points = X.shape[-1]  # num of points in each shape
        enc = encoding.unsqueeze(-1).repeat(1, 1, num_points)
        X_enc = torch.cat([X, enc], 1)
        X_enc = F.leaky_relu(self.fc1(X_enc))
        X_enc = F.leaky_relu(self.fc2(X_enc))

        # Return the drift from obj X determined by the latent encoding
        return X + self.fc3(X_enc)


class EnsembleAutoDecoder(nn.Module):
    """
    Ingests the latent encoding of two 3D objects
    and outputs the similarity score using an ensemble of CompNets
    Stacked Ensemble
    """

    def __init__(self, adnet_list, encoding_size=256, point_dim=3):
        """
        if comp_net is a module, EnsembleCompNet creates num_ensemble*comp_net NN modules
        if comp_net is a list of modules, EnsembleCompNet iterates through comp_net to get the NN modules
        """
        super(EnsembleAutoDecoder, self).__init__()
        self.ensemble_adnet = nn.ModuleList()

        for adnet in adnet_list:
            self.ensemble_adnet.append(adnet)

        self.fc_final = nn.Conv1d(64 * len(adnet_list), point_dim, 1)

    def forward(self, X, encoding):
        num_points = X.shape[-1]  # num of points in each shape
        enc = encoding.unsqueeze(-1).repeat(1, 1, num_points)

        X_enc = torch.cat([X, enc], 1)
        enc_output_list = [net(X_enc) for net in self.ensemble_adnet]

        # Return the drift from obj X determined by the latent encoding
        return X + self.fc_final(torch.cat(enc_output_list, dim=1))


class CompNet(nn.Module):
    """
    Ingests the latent encoding of two 3D objects
    and outputs the similarity score
    """

    def __init__(self, encoding_size=256):
        super(CompNet, self).__init__()
        self.fc1 = nn.Linear(encoding_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, encoding):
        X = F.leaky_relu(self.fc1(encoding))
        return torch.sigmoid(self.fc2(X))


class EnsembleCompNet(nn.Module):
    """
    Ingests the latent encoding of two 3D objects
    and outputs the similarity score using an ensemble of CompNets
    Stacked Ensemble
    """

    def __init__(self, comp_net=CompNet, num_ensemble=5, encoding_dim=256, seed_val=17 * 19, use_cuda=True):
        """
        if comp_net is a module, EnsembleCompNet creates num_ensemble*comp_net NN modules
        if comp_net is a list of modules, EnsembleCompNet iterates through comp_net to get the NN modules
        """
        super(EnsembleCompNet, self).__init__()
        self.ensemble_compnet = nn.ModuleList()

        if isinstance(comp_net, list):
            if num_ensemble != len(comp_net):
                raise IndexError(
                    f"Length of comp_nets: {len(comp_net)} and num_ensemble: {num_ensemble} do not match")
            comp_net_list = comp_net
            for i in range(num_ensemble):
                self.ensemble_compnet.append(comp_net_list[i])
        else:
            for i in range(num_ensemble):
                torch.manual_seed(seed_val * i + 1)
                if use_cuda:
                    torch.cuda.manual_seed(seed_val * i + 1)
                self.ensemble_compnet.append(comp_net(encoding_dim))
        self.final = nn.Linear(num_ensemble, 1)

    def forward(self, encoding):
        """ Returns the final value of the results after a nn.Linear layer """
        total_pred = torch.cat([net(encoding)
                                for net in self.ensemble_compnet])
        total_pred = total_pred.reshape(-1, len(self.ensemble_compnet))

        return torch.sigmoid(self.final(total_pred))
