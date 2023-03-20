"""
    CompletionFormer
    ======================================================================

    CompletionFormerMetric implementation
"""


import torch
from . import BaseMetric


class CompletionFormerMetric(BaseMetric):
    def __init__(self, args):
        super(CompletionFormerMetric, self).__init__(args)

        self.args = args
        self.t_valid = 0.0001

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110'
        ]

    def evaluate(self, sample, output, mode=None):
        with torch.no_grad():
            pred = output['pred'].detach()
            gt = sample['gt'].detach()

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = gt > self.t_valid
            num_valid = mask.sum()

            pred = pred[mask]
            gt = gt[mask]

            pred_inv = pred_inv[mask]
            gt_inv = gt_inv[mask]

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25**2).type_as(ratio)
            del_3 = (ratio < 1.25**3).type_as(ratio)
            del_102 = (ratio < 1.02).type_as(ratio)
            del_105 = (ratio < 1.05).type_as(ratio)
            del_110 = (ratio < 1.10).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)
            del_102 = del_102.sum() / (num_valid + 1e-8)
            del_105 = del_105.sum() / (num_valid + 1e-8)
            del_110 = del_110.sum() / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3, del_102, del_105, del_110]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result
