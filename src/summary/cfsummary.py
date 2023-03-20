"""
    CompletionFormer
    ======================================================================

    CompletionFormerSummary implementation
"""


from . import BaseSummary
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

cmap = 'jet'
cm = plt.get_cmap(cmap)


class CompletionFormerSummary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] " \
            "but got {}".format(mode)

        super(CompletionFormerSummary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def update(self, global_step, sample, output):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.5f}  ".format(name, val)]

                if (idx + 1) % 12 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        dep = sample['dep'].detach().data.cpu().numpy()
        gt = sample['gt'].detach().data.cpu().numpy()
        pred = output['pred'].detach().data.cpu().numpy()
        preds = [d.detach().data.cpu().numpy() for d in output['pred_inter']]

        if output['confidence'] is not None:
            confidence = output['confidence']
            if isinstance(confidence, (list, tuple)):
                confidence = [c.data.cpu().numpy() for c in confidence]
                if len(confidence) == 1:
                    confidence = confidence * len(preds)
            else:
                confidence = confidence.data.cpu().numpy()
                confidence = [confidence, ] * len(preds)

        else:
            confidence = [np.zeros_like(dep)] * len(preds)

        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

            rgb = rgb[0:num_summary, :, :, :]
            dep = dep[0:num_summary, :, :, :]
            gt = gt[0:num_summary, :, :, :]
            pred = pred[0:num_summary, :, :, :]
            confidence = [c[0:num_summary, :, :, :] for c in confidence]
            preds = [d[0:num_summary, :, :, :] for d in preds]

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)
        confidence = [np.clip(conf, a_min=0, a_max=1.0) for conf in confidence]

        list_img = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            dep_tmp = dep[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :]
            confidence_tmp = [conf[b, 0, :, :] for conf in confidence]
            preds_tmp = [d[b, 0, :, :] for d in preds]
            norm = plt.Normalize(vmin=gt_tmp.min(), vmax=gt_tmp.max())
            error_tmp = depth_err_to_colorbar(pred_tmp, gt_tmp)

            props = []
            for pd_tmp in preds_tmp:
                err = np.concatenate([cm(norm(pd_tmp))[..., :3], depth_err_to_colorbar(pd_tmp, gt_tmp)], axis=1)
                err = np.transpose(err[:, :, :3], (2, 0, 1))
                props.append(err)

            props = np.concatenate(props, axis=1)

            conf_norm = plt.Normalize(vmin=0, vmax=1)
            confs = []
            num_conf = len(confidence_tmp)
            for conf_tmp in confidence_tmp:
                conf = plt.get_cmap('gray')(conf_norm(conf_tmp))
                conf = np.transpose(conf[:, :, :3], (2, 0, 1))
                confs.append(conf)
            confidence_tmp = confs[0]
            confs = np.concatenate(confs, axis=1)
            if len(preds_tmp) == num_conf:
                props = np.concatenate([props, confs], axis=2)

            self.add_image(self.mode + '/props_{}'.format(b), props, global_step)

            dep_tmp = cm(norm(dep_tmp))
            gt_tmp = cm(norm(gt_tmp))
            pred_tmp = cm(norm(pred_tmp))

            dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            error_tmp = np.transpose(error_tmp[:, :, :3], (2, 0, 1))

            img = np.concatenate((rgb_tmp, dep_tmp, pred_tmp, gt_tmp, error_tmp,
                                  confidence_tmp), axis=1)

            list_img.append(img)

        img_total = np.concatenate(list_img, axis=2)
        img_total = torch.from_numpy(img_total)

        self.add_image(self.mode + '/images', img_total, global_step)

        self.flush()

        # Reset
        self.loss = []
        self.metric = []

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                                                              self.mode, epoch)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_pred = '{}/{:010d}.png'.format(self.path_output, idx)

                pred = output['pred'].detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()

                pred = (pred*256.0).astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)
            else:
                # Parse data
                feat_init = output['pred_init']
                list_feat = output['pred_inter']

                rgb = sample['rgb'].detach()
                dep = sample['dep'].detach()
                pred = output['pred'].detach()
                gt = sample['gt'].detach()

                pred = torch.clamp(pred, min=0)

                # Un-normalization
                rgb.mul_(self.img_std.type_as(rgb)).add_(
                    self.img_mean.type_as(rgb))

                rgb = rgb[0, :, :, :].data.cpu().numpy()
                dep = dep[0, 0, :, :].data.cpu().numpy()
                pred = pred[0, 0, :, :].data.cpu().numpy()
                pred_gray = pred
                gt = gt[0, 0, :, :].data.cpu().numpy()
                feat_init = feat_init[0, 0, :, :].data.cpu().numpy()
                max_depth = max(gt.max(), pred.max())
                norm = plt.Normalize(vmin=gt.min(), vmax=gt.max())

                rgb = np.transpose(rgb, (1, 2, 0))
                for k in range(0, len(list_feat)):
                    feat_inter = list_feat[k]
                    feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()
                    feat_inter = np.concatenate((rgb, cm(norm(pred))[...,:3], cm(norm(gt))[...,:3], depth_err_to_colorbar(feat_inter, gt)), axis=0)

                    list_feat[k] = feat_inter

                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(
                    self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_rgb = '{}/01_rgb.png'.format(self.path_output)
                path_save_dep = '{}/02_dep.png'.format(self.path_output)
                path_save_init = '{}/03_pred_init.png'.format(self.path_output)
                path_save_pred = '{}/05_pred_final.png'.format(self.path_output)
                path_save_pred_gray = '{}/05_pred_final_gray.png'.format(self.path_output)
                path_save_gt = '{}/06_gt.png'.format(self.path_output)
                path_save_error = '{}/07_error.png'.format(self.path_output)

                plt.imsave(path_save_rgb, rgb, cmap=cmap)
                plt.imsave(path_save_gt, cm(norm(gt)))
                plt.imsave(path_save_pred, cm(norm(pred)))
                plt.imsave(path_save_pred_gray, pred_gray, cmap='gray')
                plt.imsave(path_save_dep, cm(norm(dep)))
                plt.imsave(path_save_init, cm(norm(feat_init)))
                plt.imsave(path_save_error, depth_err_to_colorbar(pred, gt, with_bar=True))

                for k in range(0, len(list_feat)):
                    path_save_inter = '{}/04_pred_prop_{:02d}.png'.format(
                        self.path_output, k)
                    plt.imsave(path_save_inter, list_feat[k])



def depth_err_to_colorbar(est, gt=None, with_bar=False, cmap='jet'):
    error_bar_height = 50
    if gt is None:
        gt = np.zeros_like(est)
        valid = est > 0
        max_depth = est.max()
    else:
        valid = gt > 0
        max_depth = gt.max()
    error_map = np.abs(est - gt) * valid
    h, w= error_map.shape

    maxvalue = error_map.max()
    if max_depth < 30:
        breakpoints = np.array([0,      0.1,      0.5,      1.25,     2,    4,       max(10, maxvalue)])
    else:
        breakpoints = np.array([0,      0.1,      0.5,      1.25,     2,    4,     max(90, maxvalue)])
    points      = np.array([0,      0.25,   0.38,   0.66,  0.83,  0.95,     1])
    num_bins    = np.array([0,      w//8,   w//8,   w//4,  w//4,  w//8,     w - (w//4 + w//4 + w//8 + w//8 + w//8)])
    acc_num_bins = np.cumsum(num_bins)

    for i in range(1, len(breakpoints)):
        scale = points[i] - points[i-1]
        start = points[i-1]
        lower = breakpoints[i-1]
        upper = breakpoints[i]
        error_map = revalue(error_map, lower, upper, start, scale)

    # [0, 1], [H, W, 3]
    error_map = plt.cm.get_cmap(cmap)(error_map)[:, :, :3]

    if not with_bar:
        return error_map

    error_bar = np.array([])
    for i in range(1, len(num_bins)):
        error_bar = np.concatenate((error_bar, np.linspace(points[i-1], points[i], num_bins[i])))

    error_bar = np.repeat(error_bar, error_bar_height).reshape(w, error_bar_height).transpose(1, 0) # [error_bar_height, w]
    error_bar_map = plt.cm.get_cmap(cmap)(error_bar)[:, :, :3]
    plt.xticks(ticks=acc_num_bins, labels=[str(f) for f in breakpoints])
    plt.axis('on')

    # [0, 1], [H, W, 3]
    error_map = np.concatenate((error_map, error_bar_map[..., :3]), axis=0)[..., :3]

    return error_map

def revalue(map, lower, upper, start, scale):
    mask = (map > lower) & (map <= upper)
    if np.sum(mask) >= 1.0:
        mn, mx = map[mask].min(), map[mask].max()
        map[mask] = ((map[mask] - mn) / (mx -mn + 1e-7)) * scale + start

    return map

