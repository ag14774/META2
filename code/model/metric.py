import torch
import torch.nn as nn
from base.base_metric import BaseMetric
from model.util import convert_batch_to_prob, write_prob_dist


class AccuracyTopK(BaseMetric):
    def __init__(self, k=3):
        super(AccuracyTopK, self).__init__()
        self.k = k

    def forward(self, output, target):
        with torch.no_grad():
            pred = torch.topk(output, self.k, dim=1)[1]
            assert pred.shape[0] == len(target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(pred[:, i] == target).item()
        return correct / len(target)

    def labels(self):
        return [f"{self.__class__.__name__}.{self.k}"]


class AccuracyPerTaxGroup(BaseMetric):
    def __init__(
        self,
        k=1,
        groups=['phylum', 'class', 'order', 'family', 'genus', 'species'],
        selected_levels=None,
        mil_mode=False):
        super(AccuracyPerTaxGroup, self).__init__()
        self.k = k
        self.groups = [*groups, 'leaf']
        if not selected_levels:
            selected_levels = self.groups

        self.selected_levels = selected_levels
        self.level2idx = {l: i for i, l in enumerate(self.groups)}
        self.selected_levels_idx = [self.level2idx[i] for i in selected_levels]
        self.mil_mode = mil_mode

    def compute_counts_levels(self, predictions, target):
        with torch.no_grad():
            ac = torch.zeros(len(predictions))
            for group in range(len(predictions)):
                for k in range(predictions[group].size(-1)):
                    ac[group] += torch.sum(
                        predictions[group][:, k] ==
                        target[:, self.selected_levels_idx[group]]).item()
        return ac

    def forward(self, output, target):
        with torch.no_grad():
            if self.mil_mode:
                target = target.view(-1, *target.size()[2:])
            if not isinstance(output, list):
                output = [output]
            output = [torch.topk(yp, self.k, dim=1)[1] for yp in output]
            ac = self.compute_counts_levels(output, target) / target.size(0)
        # print(ac.device, output[0].device, target.device)
        # print(ac.shape)
        return ac

    def labels(self):
        return [f"Accuracy.{self.k}.{g}" for g in self.selected_levels]


class JSPerTaxGroup(BaseMetric):
    def __init__(
        self,
        model_out_format='probs',
        target_out_format='probs',
        groups=['phylum', 'class', 'order', 'family', 'genus', 'species'],
        selected_levels=None,
        add_figures=True):
        super(JSPerTaxGroup, self).__init__()
        self.groups = [*groups, 'leaf']
        if not selected_levels:
            selected_levels = self.groups
        self.selected_levels = selected_levels
        self.level2idx = {l: i for i, l in enumerate(self.groups)}
        self.selected_levels_idx = [self.level2idx[i] for i in selected_levels]

        self.KL = nn.KLDivLoss(reduction='batchmean')
        assert model_out_format in ['probs', 'logprobs', 'counts']
        self.model_out_format = model_out_format
        assert target_out_format in ['probs', 'logprobs', 'counts']
        self.target_out_format = target_out_format
        self.add_figures = add_figures

    def forward(self, outputs, targets):
        with torch.no_grad():
            if not isinstance(outputs, list):
                outputs = [outputs]
            res = torch.zeros(len(outputs))

            for i, (output,
                    group) in enumerate(zip(outputs, self.selected_levels)):
                new_target = targets[self.selected_levels_idx[i]]
                outprobs = output

                if self.model_out_format == 'logprobs':
                    outprobs = torch.exp(outprobs)
                elif self.model_out_format == 'counts':
                    outprobs = outprobs + 0.001
                    outprobs = outprobs / torch.sum(
                        outprobs, dim=1, keepdim=True)

                if self.target_out_format == 'logprobs':
                    new_target = torch.exp(new_target)
                elif self.target_out_format == 'counts':
                    new_target = new_target + 0.001
                    new_target = new_target / torch.sum(
                        new_target, dim=1, keepdim=True)

                if self.add_figures:
                    write_prob_dist(outprobs[0], new_target[0], group)
                avg_distr = (outprobs + new_target) / 2
                loss1 = self.KL(torch.log(outprobs), avg_distr).item()
                loss2 = self.KL(torch.log(new_target), avg_distr).item()
                res[i] = (loss1 + loss2) / 2

        return res

    def labels(self):
        return [f"JS.{g}" for g in self.selected_levels]


class JSPerTaxGroupWithCounts(BaseMetric):
    def __init__(
        self,
        groups=['phylum', 'class', 'order', 'family', 'genus', 'species'],
        selected_levels=None,
        add_figures=True,
        mil_mode=False):
        super().__init__()
        self.groups = [*groups, 'leaf']
        if not selected_levels:
            selected_levels = self.groups
        self.selected_levels = selected_levels
        self.level2idx = {l: i for i, l in enumerate(self.groups)}
        self.selected_levels_idx = [self.level2idx[i] for i in selected_levels]

        self.KL = nn.KLDivLoss(reduction='sum')
        self.add_figures = add_figures
        self.mil_mode = mil_mode

    def forward(self, outputs, targets):
        if self.mil_mode:
            return self.forward_mil(outputs, targets)
        else:
            return self.forward_non_mil(outputs, targets)

    def forward_non_mil(self, outputs, targets):
        with torch.no_grad():
            if not isinstance(outputs, list):
                outputs = [outputs]
            res = torch.zeros(len(outputs))
            targets = targets.t()
            for i, (output,
                    group) in enumerate(zip(outputs, self.selected_levels)):
                target = targets[self.selected_levels_idx[i]]

                new_target = convert_batch_to_prob(output.size(1),
                                                   target,
                                                   eps=0.01)

                output = output + 0.01
                new_output = output.softmax(dim=1)
                new_output = torch.mean(new_output, dim=0)

                if self.add_figures:
                    write_prob_dist(new_output, new_target, group)
                avg_distr = (new_output + new_target) / 2
                loss1 = self.KL(torch.log(new_output), avg_distr).item()
                loss2 = self.KL(torch.log(new_target), avg_distr).item()
                res[i] = (loss1 + loss2) / 2

        return res

    def forward_mil(self, outputs, targets):
        with torch.no_grad():
            if not isinstance(outputs, list):
                outputs = [outputs]
            res = torch.zeros(len(outputs))
            targets = targets.transpose(1, 2)
            for i, (output,
                    group) in enumerate(zip(outputs, self.selected_levels)):
                res_tmp = 0
                for batch in targets:
                    target = batch[self.selected_levels_idx[i]]

                    new_target = convert_batch_to_prob(output.size(1),
                                                       target,
                                                       eps=0.01)

                    output = output + 0.01
                    new_output = output.softmax(dim=1)
                    new_output = torch.mean(new_output, dim=0)

                    if self.add_figures:
                        write_prob_dist(new_output, new_target, group)
                    avg_distr = (new_output + new_target) / 2
                    loss1 = self.KL(torch.log(new_output), avg_distr).item()
                    loss2 = self.KL(torch.log(new_target), avg_distr).item()
                    res_tmp += (loss1 + loss2) / 2
                res[i] = res_tmp / targets.size(0)

        return res

    def labels(self):
        return [f"JS.{g}" for g in self.selected_levels]
