import logging
import os

import numpy as np
import pandas as pd
import torch
from skimage import io

from utils import format_string, convert_from_color, count_sliding_window, grouper, sliding_window, CrossEntropy2d, dice_loss, \
    metrics, convert_to_color

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def test(dataset_cfg, training_cfg, model, test_ids, all=False, test_loader=None):
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen':
        stride = dataset_cfg.stride_size
    batch_size = training_cfg.batch_size
    window_size = tuple(training_cfg.window_size)
    N_CLASSES = dataset_cfg.n_classes
    # Use the network on the test set
    if dataset_cfg.name == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32')
                       for id in
                       test_ids)
    elif dataset_cfg.name == 'Vaihingen':
        test_images = (1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32') for id in
                       test_ids)

    if dataset_cfg.name == 'Potsdam':
        dif_ids = [format_string(id) for id in test_ids]
        test_dsms = (np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32') for id in dif_ids)
    else:
        test_dsms = (np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32') for id in test_ids)

    invert_palette = {tuple(v): k for k, v in dataset_cfg.palette.items()}
    test_labels = (convert_from_color(io.imread(dataset_cfg.label_folder.format(id)), invert_palette) for id in
                   test_ids)
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen':
        eroded_labels = (convert_from_color(io.imread(dataset_cfg.eroded_folder.format(id)), invert_palette) for id in
                         test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen':
        with torch.no_grad():
            for img, dsm, gt, gt_e in zip(test_images, test_dsms, test_labels, eroded_labels):
                pred = np.zeros(img.shape[:2] + (N_CLASSES,))

                total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
                for i, coords in enumerate(
                        grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                    # Build the tensor
                    image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = torch.from_numpy(image_patches).cuda()

                    min = np.min(dsm)
                    max = np.max(dsm)
                    dsm = (dsm - min) / (max - min)
                    dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                    dsm_patches = np.asarray(dsm_patches)
                    dsm_patches = torch.from_numpy(dsm_patches).cuda()

                    # Do the inference
                    outs, _, _ = model(image_patches, dsm_patches)
                    outs = outs.data.cpu().numpy()

                    # Fill in the results array
                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1, 2, 0))
                        pred[x:x + w, y:y + h] += out
                    del (outs)

                pred = np.argmax(pred, axis=-1)
                all_preds.append(pred)
                all_gts.append(gt_e)
                # clear_output()

        results = metrics(np.concatenate([p.ravel() for p in all_preds]),
                          np.concatenate([p.ravel() for p in all_gts]).ravel(), dataset_cfg.labels,
                          dataset_cfg.n_classes)

        if all:
            return results, all_preds, all_gts
        else:
            return results

    elif dataset_cfg.name == 'WHU' or dataset_cfg.name == 'YESeg':
        with torch.no_grad():
            for img, dsm, gt in test_loader:
                img, dsm = img.cuda(), dsm.cuda()
                outs, _, _ = model(img, dsm)
                outs = outs.data.cpu().numpy()

                pred = np.argmax(outs, axis=1)
                all_preds.append(pred)
                all_gts.append(gt)

        results = metrics(np.concatenate([p.ravel() for p in all_preds]),
                          np.concatenate([p.ravel() for p in all_gts]).ravel(), dataset_cfg.labels,
                          dataset_cfg.n_classes)

        if all:
            return results, all_preds, all_gts
        else:
            return results


def train(dataset_cfg, training_cfg, model, optimizer, scheduler, train_loader, weights, results_dir, test_loader=None):
    weights = weights.cuda()
    epochs = training_cfg.epochs
    save_epoch = training_cfg.save_epoch

    history = {
        'round': [],
        'train_loss': [],
        'Kappa': [],
        'OA_total': [],
        'MIoU_mean': [],
        'F1_mean': []
    }

    for label in dataset_cfg.labels:
        history[f'OA_{label}'] = []
        history[f'MIoU_{label}'] = []
        history[f'F1_{label}'] = []

    MIoU_best = 0.0
    epoch_best = -1
    for epoch in range(1, epochs + 1):
        logger.info('Train (epoch {}/{})'.format(epoch, epochs))
        model.train()
        batch_losses = []
        for batch_idx, (opt, dsm, target) in enumerate(train_loader):
            opt, dsm, target = opt.cuda(), dsm.cuda(), target.cuda()
            optimizer.zero_grad()

            output, MIloss, low_MIloss = model(opt, dsm)
            loss_ce = CrossEntropy2d(output, target, weight=weights)
            loss_dice = dice_loss(output, target)
            loss = loss_ce + (MIloss * training_cfg.alpha) - (low_MIloss * training_cfg.beta) + (loss_dice * training_cfg.gamma)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            batch_losses.append(loss.item())

            del (opt, target, loss)

        epoch_loss = np.mean(batch_losses)

        if epoch % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            model.eval()
            if dataset_cfg.name == 'WHU' and test_loader is not None:
                results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False,
                                   test_loader=test_loader)
            elif dataset_cfg.name == 'YESeg' and test_loader is not None:
                results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False,
                                   test_loader=test_loader)
            else:
                results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False)
            model.train()

            MIoU = results_val['MIoU']['mean']

            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(results_val['Kappa'])

            history['OA_total'].append(results_val['OA']['total'])
            for i in dataset_cfg.labels:
                history['OA_{}'.format(i)].append(results_val['OA'][i])

            history['MIoU_mean'].append(results_val['MIoU']['mean'])
            for i in dataset_cfg.labels:
                history['MIoU_{}'.format(i)].append(results_val['MIoU'][i])

            history['F1_mean'].append(results_val['F1']['mean'])
            for i in dataset_cfg.labels:
                history['F1_{}'.format(i)].append(results_val['F1'][i])

            if MIoU > MIoU_best:
                if dataset_cfg.name == 'Vaihingen':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_vaihingen'))
                elif dataset_cfg.name == 'Potsdam':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_potsdam'))
                elif dataset_cfg.name == 'WHU':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_whu'))
                elif dataset_cfg.name == 'YESeg':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_yeseg'))

                MIoU_best = MIoU
                epoch_best = epoch

            logger.info('    Training Loss: {}'.format(epoch_loss))
            logger.info('    Kappa: {}'.format(results_val["Kappa"]))
            logger.info('    OA: {}'.format(results_val["OA"]))
            logger.info('    F1: {}'.format(results_val["F1"]))
            logger.info('    MIoU: {}'.format(results_val["MIoU"]))
            logger.info("")
        else:
            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(0.0)

            history['OA_total'].append(0.0)
            for i in dataset_cfg.labels:
                history['OA_{}'.format(i)].append(0.0)

            history['MIoU_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history['MIoU_{}'.format(i)].append(0.0)

            history['F1_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history['F1_{}'.format(i)].append(0.0)

            logger.info('    Training Loss: {}'.format(epoch_loss))

    logger.info('Best epoch {}, MIoU best: {}'.format(epoch_best, MIoU_best))

    df = pd.DataFrame(history)
    if dataset_cfg.name == 'Vaihingen':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_vaihingen'))
    elif dataset_cfg.name == 'Potsdam':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_potsdam'))
    elif dataset_cfg.name == 'WHU':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_whu'))
    elif dataset_cfg.name == 'YESeg':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_yeseg'))

    logger.info('End of training !')


def visualize_testloader(model, test_loader, palette, save_root):
    # os.makedirs(save_root, exist_ok=True)
    model.cuda()
    model.eval()
    tile_idx = 0
    with torch.no_grad():
        for img, dsm, _ in test_loader:
            img, dsm = img.cuda(), dsm.cuda()
            pred, _, _ = model(img, dsm)
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            for i in range(pred.shape[0]):
                color_pred = convert_to_color(pred[i], palette)
                io.imsave(os.path.join(save_root, f"tile_{tile_idx}.png"),
                          color_pred, check_contrast=False)
                tile_idx += 1
