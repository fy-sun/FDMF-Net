import logging
import os
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from skimage import io

from models import Baseline
from train import train, test, visualize_testloader
from utils import ISPRS_dataset, convert_to_color, fix_random_seed, WHUDataset

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler(),                                # 输出到控制台
#         logging.FileHandler("my_log.log", mode="w")     # 同时输出到文件
#     ]
# )


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    fix_random_seed(cfg.seed)

    logger.info("Loaded configuration:")
    # print(OmegaConf.to_yaml(cfg))
    logger.info("--- training configuration:")
    logger.info("\n%s", OmegaConf.to_yaml(cfg.training, resolve=True))
    logger.info("--- model configuration:")
    logger.info("\n%s", OmegaConf.to_yaml(cfg.model, resolve=True))

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.cuda_visible_devices))
    logger.info("Using GPUs: %s %d", os.environ["CUDA_VISIBLE_DEVICES"], len(cfg.cuda_visible_devices))
    logger.info("Using model: %s", cfg.training_model)
    logger.info("Using dataset: %s", cfg.training_dataset)

    cfg.training.batch_size = cfg.training.batch_size * len(cfg.cuda_visible_devices)
    cfg.training.learning_rate = cfg.training.learning_rate * len(cfg.cuda_visible_devices)
    if cfg.training_dataset == 'WHU':
        cfg.training.learning_rate = 1e-4

    logger.info("Real Training Configuration:")
    logger.info("batch size: %d", cfg.training.batch_size)
    logger.info("learning rate: %f", cfg.training.learning_rate)

    dataset_cfg = cfg.dataset.datasets[cfg.training_dataset]
    N_CLASSES = dataset_cfg.n_classes
    WEIGHTS = torch.ones(N_CLASSES)

    train_ids = dataset_cfg.train_ids
    test_ids = dataset_cfg.test_ids

    if cfg.training_model == 'Baseline':
        model = Baseline(
            dataset=cfg.training_dataset,
            num_classes=N_CLASSES,
            pretrained=cfg.model.pretrained,
            model_dim=cfg.model.model_dim,
            num_heads=cfg.model.num_heads,
            ffn_dim=cfg.model.ffn_dim,
            dropout=cfg.model.dropout,
            num_layers=cfg.model.num_layers,
        )
    else:
        raise NotImplementedError("Model {} is not implemented".format(cfg.training_model))

    model = model.cuda()
    model = nn.DataParallel(model)

    total_params = sum(param.nelement() for param in model.parameters())
    logger.info('All Params: %d', total_params)

    logger.info("training : %s", train_ids)
    logger.info("testing : %s", test_ids)

    if cfg.training_dataset == 'WHU':
        train_dataset = WHUDataset(
            data_dir=os.path.join(cfg.folder, 'OPT_SAR_WHU_crops'),
            split='train',
            color_map={k: tuple(v) for k, v in dataset_cfg.palette.items()}
        )
        test_dataset = WHUDataset(
            data_dir=os.path.join(cfg.folder, 'OPT_SAR_WHU_crops'),
            split='test',
            color_map={k: tuple(v) for k, v in dataset_cfg.palette.items()}
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                                   num_workers=cfg.training.num_workers, pin_memory=True,
                                                   worker_init_fn=lambda worker_id: fix_random_seed(
                                                       cfg.seed + worker_id))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                                                  num_workers=cfg.training.num_workers, pin_memory=True,
                                                  worker_init_fn=lambda worker_id: fix_random_seed(
                                                      cfg.seed + worker_id))
    else:
        train_dataset = ISPRS_dataset(train_ids, dataset_cfg, cfg.training.window_size, cache=cfg.training.cache,
                                      augmentation=cfg.training.augmentation)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                                   num_workers=cfg.training.num_workers, pin_memory=True,
                                                   worker_init_fn=lambda worker_id: fix_random_seed(
                                                       cfg.seed + worker_id))

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate,
                                weight_decay=cfg.training.weight_decay, momentum=cfg.training.momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.training.milestones, gamma=0.1)

    # if cfg.training_dataset == 'Vaihingen':
    #     results_dir = os.path.join(os.getcwd(), 'results_{}_vaihingen'.format(cfg.training_model))
    # elif cfg.training_dataset == 'Potsdam':
    #     results_dir = os.path.join(os.getcwd(), 'results_{}_potsdam'.format(cfg.training_model))
    # elif cfg.training_dataset == 'WHU':
    #     results_dir = os.path.join(os.getcwd(), 'results_{}_whu'.format(cfg.training_model))
    # else:
    #     raise ValueError("Unsupported dataset: " + cfg.training_dataset)
    # os.makedirs(results_dir, exist_ok=True)
    # logger.info("Experiment output directory: %s", os.getcwd())

    results_dir = os.path.join('Baseline_WHU_42/2025-05-21_22-16-15',
                               'results_{}_{}'.format(cfg.training_model, cfg.training_dataset))

    # start_train = time.time()
    # logger.info('Start training...')
    # if cfg.training_dataset == 'WHU':
    #     train(dataset_cfg, cfg.training, model, optimizer, scheduler, train_loader, WEIGHTS, results_dir, test_loader=test_loader)
    # else:
    #     train(dataset_cfg, cfg.training, model, optimizer, scheduler, train_loader, WEIGHTS, results_dir)
    # end_train = time.time()
    # logger.info('Training time: {:.2f} hours'.format((end_train - start_train) / 3600))
    # logger.info("")

    start_test = time.time()
    logger.info('Start testing...')
    palette = {k: tuple(v) for k, v in dataset_cfg.palette.items()}
    # test
    if cfg.training_dataset == 'Vaihingen':
        # best model eval
        best_model_path = os.path.join(results_dir, 'best_model_vaihingen')
        model.load_state_dict(torch.load(best_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True)
        logger.info('Best model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p, palette)
            io.imsave(os.path.join(results_dir, 'best_model_inference_{}_tile_{}.png'.format(cfg.training_model, id_)),
                      img)
        logger.info('-----------------------------------------')
        # final model eval
        final_model_path = os.path.join(results_dir, 'final_model_vaihingen')
        model.load_state_dict(torch.load(final_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True)
        logger.info('Final model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p, palette)
            io.imsave(os.path.join(results_dir, 'final_model_inference_{}_tile_{}.png'.format(cfg.training_model, id_)),
                      img)

    elif cfg.training_dataset == 'Potsdam':
        best_model_path = os.path.join(results_dir, 'best_model_potsdam')
        model.load_state_dict(torch.load(best_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True)
        logger.info('Best model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p, palette)
            io.imsave(os.path.join(results_dir, 'best_model_inference_{}_tile_{}.png'.format(cfg.training_model, id_)),
                      img)
        logger.info('-----------------------------------------')
        final_model_path = os.path.join(results_dir, 'final_model_potsdam')
        model.load_state_dict(torch.load(final_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True)
        logger.info('Final model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p, palette)
            io.imsave(os.path.join(results_dir, 'final_model_inference_{}_tile_{}.png'.format(cfg.training_model, id_)),
                      img)

    elif cfg.training_dataset == 'WHU':
        best_model_path = os.path.join(results_dir, 'best_model_whu')
        model.load_state_dict(torch.load(best_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True,
                                           test_loader=test_loader)
        logger.info('Best model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        # for p, id_ in zip(all_preds, range(len(test_loader))):
        #     img = convert_to_color(p, palette)
        #     io.imsave(os.path.join(results_dir, 'best_model_inference_{}_tile_{}.png'.format(cfg.training_model, id_)),
        #               img)
        logger.info('-----------------------------------------')
        final_model_path = os.path.join(results_dir, 'final_model_whu')
        model.load_state_dict(torch.load(final_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True,
                                           test_loader=test_loader)
        logger.info('Final model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        visualize_testloader(model, test_loader, palette, results_dir)

    end_test = time.time()
    logger.info('Testing time: {:.2f} hours'.format((end_test - start_test) / 3600))


if __name__ == '__main__':
    main()
