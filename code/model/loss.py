import torch
import torch.nn as nn
import numpy as np
from model.util import (convert_batch_to_prob, convert_batch_to_log_prob,
                        convert_batch_to_counts)


class CrossEntropyLossPerRank(nn.Module):
    def __init__(self,
                 selected_levels=None,
                 all_levels=[
                     'phylum', 'class', 'order', 'family', 'genus', 'species',
                     'leaf'
                 ],
                 class_percentages=None,
                 mil_mode=False):
        super().__init__()
        if not selected_levels:
            selected_levels = all_levels
        self.level2idx = {l: i for i, l in enumerate(all_levels)}
        self.selected_levels_idx = [self.level2idx[i] for i in selected_levels]
        num_of_layers = len(selected_levels)

        if class_percentages is None:
            class_percentages = []

        final_weights = []
        if class_percentages:
            for j in self.selected_levels_idx:
                perc = class_percentages[j]
                perc = 1.0 / perc.astype(np.float32)
                final_weights.append(torch.as_tensor(perc / np.sum(perc)))

        self.cross_entropies = nn.ModuleList()
        for i in range(num_of_layers):
            if final_weights:
                self.cross_entropies.append(
                    nn.CrossEntropyLoss(weight=final_weights[i],
                                        reduction='none'))
            else:
                self.cross_entropies.append(
                    nn.CrossEntropyLoss(reduction='none'))

        self.mil_mode = mil_mode

    def forward(self, outputs, targets):
        '''
        Args:
            outputs (list): list of length L of (N, C_l) tensors.
            L softmax layers
            targets (tensor): tensor of size (N, L)
            class_percentages (list): list of L arrays
        '''
        if self.mil_mode:
            targets = targets.view(-1, *targets.size()[2:])
        targets = targets.t()

        res = 0
        if not isinstance(outputs, list):
            outputs = [outputs]
        for idx, output, cross_entropy in zip(self.selected_levels_idx,
                                              outputs, self.cross_entropies):
            target = targets[idx]
            # print("output", output, "target", target)
            loss = cross_entropy(output, target)
            res = res + loss

        res = torch.mean(res)
        return res


class JSPerRank(nn.Module):
    def __init__(self,
                 selected_levels=None,
                 all_levels=[
                     'phylum', 'class', 'order', 'family', 'genus', 'species',
                     'leaf'
                 ]):
        super().__init__()
        if not selected_levels:
            selected_levels = all_levels
        self.level2idx = {l: i for i, l in enumerate(all_levels)}
        self.selected_levels_idx = [self.level2idx[i] for i in selected_levels]
        self.KL = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, targets):
        '''
        Args:
            outputs (list): list of length L of (N, C_l) tensors.
            L softmax layers
            targets (tensor): tensor of size (N, L)
            class_percentages (list): list of L arrays
        '''
        res = 0
        if not isinstance(outputs, list):
            outputs = [outputs]
        for idx, output in zip(self.selected_levels_idx, outputs):
            new_target = targets[idx]
            # print("Output:", output, "Target:", new_target)
            avg_distr = (new_target + torch.exp(output)) / 2
            loss1 = self.KL(output, avg_distr)
            loss2 = self.KL(torch.log(new_target), avg_distr)
            loss = 0.5 * loss1 + 0.5 * loss2
            res = res + loss

        res = res / len(outputs)
        return res


class MSEPerRank(nn.Module):
    def __init__(self,
                 class_percentages=None,
                 selected_levels=None,
                 all_levels=[
                     'phylum', 'class', 'order', 'family', 'genus', 'species',
                     'leaf'
                 ]):
        super().__init__()
        if not selected_levels:
            selected_levels = all_levels
        self.level2idx = {l: i for i, l in enumerate(all_levels)}
        self.selected_levels_idx = [self.level2idx[i] for i in selected_levels]

        if class_percentages is None:
            class_percentages = []
            self.num_of_weights = 0

        if class_percentages:
            self.num_of_weights = len(self.selected_levels_idx)
            for i in range(len(self.selected_levels_idx)):
                self.register_buffer(f'weights_{i}', None)

            for i, idx in enumerate(self.selected_levels_idx):
                perc = class_percentages[idx]
                perc = 1.0 / perc.astype(np.float32)
                setattr(self, f'weights_{i}',
                        torch.as_tensor(perc / np.sum(perc)))

        self.MSE = nn.MSELoss(reduction='none')

    def _get_weights(self):
        if self.num_of_weights == 0:
            return []
        final_weights = [
            getattr(self, f'weights_{i}') for i in range(self.num_of_weights)
        ]
        return final_weights

    def forward(self, outputs, targets):
        '''
        Args:
            outputs (list): list of length L of (N, C_l) tensors.
            L softmax layers
            targets (tensor): tensor of size (N, L)
            class_percentages (list): list of L arrays
        '''
        res = 0
        if not isinstance(outputs, list):
            outputs = [outputs]
        weights = self._get_weights()
        for i, output in enumerate(outputs):
            new_target = targets[self.selected_levels_idx[i]]
            # print("Output:", torch.exp(output), "Target:",
            # torch.exp(new_target))
            loss = self.MSE(output, new_target).view(-1)
            print("Unreduced loss size:", loss.size())

            if weights:
                loss = loss * weights[i]
                loss = torch.sum(loss)
            else:
                loss = torch.mean(loss)

            res = res + loss

        res = res / len(outputs)
        return res


if __name__ == '__main__':
    # loss = GeNetLoss(2)
    outputs = [
        torch.tensor([[0.7, 0.3], [0.9, 0.1]]),
        torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.7, 0.1]])
    ]
    targets = torch.tensor([[0, 2], [1, 0]])
    # print(loss)
    # print(loss(outputs, targets))

    class_percentages = [
        torch.tensor([0.5, 0.5]).float(),
        torch.tensor([1 / 3, 1 / 3, 1 / 3]).float()
    ]
    # loss = GeNetLoss(2, class_percentages)
    # print(loss(outputs, targets))
