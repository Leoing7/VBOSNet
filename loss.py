import torch
from torch import nn
from torch.nn import functional as F


class SpO2Loss(nn.Module):

    def __init__(self, lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse = nn.MSELoss()

    def pearson_loss(self, pred, target):
        v_pred = pred - torch.mean(pred)
        v_target = target - torch.mean(target)
        corr = torch.sum(v_pred * v_target) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_target ** 2)) + 1e-06)
        return 1 - corr

    def forward(self, outputs, labels, contrast_loss=0.0):
        mse_loss = self.mse(outputs, labels)
        pearson_loss = self.pearson_loss(outputs, labels)
        total_loss = self.lambda1 * mse_loss + self.lambda2 * pearson_loss + contrast_loss
        return total_loss


class MultiPositiveTripletLoss(nn.Module):

    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super(MultiPositiveTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def pairwise_distance(self, a, b):
        if self.distance_metric == 'euclidean':
            return F.pairwise_distance(a, b, p=2)
        elif self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(a, b)
        else:
            raise ValueError(f'Unsupported distance metric: {self.distance_metric}')

    def forward(self, anchor, positive1, positive2, negative):
        d_ap1 = self.pairwise_distance(anchor, positive1)
        d_ap2 = self.pairwise_distance(anchor, positive2)
        d_an = self.pairwise_distance(anchor, negative)
        avg_d_ap = (d_ap1 + d_ap2) / 2.0
        losses = F.relu(avg_d_ap - d_an + self.margin)
        return losses.mean()


class MultiPositiveContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(MultiPositiveContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08

    def forward(self, anchor, positive1, positive2, negative):
        anchor = F.normalize(anchor, p=2, dim=1)
        positive1 = F.normalize(positive1, p=2, dim=1)
        positive2 = F.normalize(positive2, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        sim_ap1 = torch.sum(anchor * positive1, dim=-1)
        sim_ap2 = torch.sum(anchor * positive2, dim=-1)
        sim_an = torch.sum(anchor * negative, dim=-1)
        all_sims = torch.cat([sim_ap1, sim_ap2, sim_an])
        mean_sim = all_sims.mean().detach()
        std_sim = all_sims.std().detach()
        sim_ap1 = (sim_ap1 - mean_sim) / (std_sim + 1e-08)
        sim_ap2 = (sim_ap2 - mean_sim) / (std_sim + 1e-08)
        sim_an = (sim_an - mean_sim) / (std_sim + 1e-08)
        logits = torch.stack([sim_ap1, sim_ap2, sim_an], dim=1) / self.temperature
        log_numerator = torch.logsumexp(logits[:, :2], dim=1)
        log_denominator = torch.logsumexp(logits, dim=1)
        losses = log_denominator - log_numerator
        return losses.mean()
