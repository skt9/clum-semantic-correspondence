import random, time
import torch
import numpy as np
from archs.model import SupervisedNet, UnsupervisedNet
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from pathlib import Path
from typing import List
import time
from SPair71k import SPair71K
from utils.utils import set_deterministic
from utils.evaluation_metric import matching_accuracy, f1_score, get_pos_neg
from utils.dataloader_utils import worker_init_fix, worker_init_rand, collate_fn
from utils.config import cfg as cfg
from utils.loss_utils import HammingLoss, CycleLoss
import torchvision.transforms as transforms
from tqdm import tqdm
from gm_dataset import GMDataset
# from utils.utils import np2torch

sets_translation_dict = dict(train="trn", test="test")

difficulty_params_dict = dict(
    trn=cfg.TRAIN.difficulty_params, val=cfg.EVAL.difficulty_params, test=cfg.EVAL.difficulty_params
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using CPU.")

def do_evaluation(model, dataset, dataloader, eval_epoch=None, verbose=False):
    print("Start evaluation...")
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / "params" / "params_{:04}.pt".format(eval_epoch))
        print("Loading model parameters from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

    was_training = model.training
    model.eval()

    ds = dataset
    ds.set_num_graphs(2)
    classes = ds.classes
    cls_cache = ds.cls

    accs = torch.zeros(len(classes), device=device)
    f1_scores = torch.zeros(len(classes), device=device)

    for i, cls in enumerate(classes):
        if verbose:
            print("Evaluating class {}: {}/{}".format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.set_cls(cls)
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        tp = torch.zeros(1, device=device)
        fp = torch.zeros(1, device=device)
        fn = torch.zeros(1, device=device)

        for k, inputs in enumerate(dataset):

            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt = [_.cuda() for _ in inputs["ns"]]
            edges = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            batch_num = data_list[0].size(0)

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                s_pred_list = model(
                    data_list,
                    points_gt,
                    edges,
                    n_points_gt,
                    perm_mat_list
                )

            _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_list[0], perm_mat_list[0])
            _tp, _fp, _fn = get_pos_neg(s_pred_list[0], perm_mat_list[0])

            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num
            tp += _tp
            fp += _fp
            fn += _fn

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print("Class {:<8} Iteration {:<4} {:>4.2f}sample/s".format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        f1_scores[i] = f1_score(tp, fp, fn)
        if verbose:
            print("Class {} acc = {:.4f} F1 = {:.4f}".format(cls, accs[i], f1_scores[i]))

    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print("Matching accuracy")
    for cls, single_acc, f1_sc in zip(classes, accs, f1_scores):
        print("{} = {:.4f}, {:.4f}".format(cls, single_acc, f1_sc))
    print("average = {:.4f}, {:.4f}".format(torch.mean(accs), torch.mean(f1_scores)))

    return accs, f1_scores

if __name__ == "__main__":

    torch.cuda.empty_cache()
    # model = UnsupervisedNet()
    model = SupervisedNet()
    model.cuda()

    feat_dim = 1024 # vitl14
    folder_path = "./data/downloaded/SPair-71k/"
    train_args = {"ds_path": folder_path, "dataset_size": "small", "mode": "trn", "num_graphs_per_instance": 2}
    tr_ds = SPair71K(**train_args)
    test_args = {"ds_path": folder_path, "dataset_size": "small", "mode": "test", "num_graphs_per_instance": 2}
    test_ds = SPair71K(**test_args)
    fix_seed = True
    tr_dataloader = torch.utils.data.DataLoader(
            tr_ds,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=False,
            worker_init_fn=worker_init_fix if fix_seed else worker_init_rand,
        )

    ds = GMDataset("SPair71k", None, **test_args)


    ts_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=False,
            worker_init_fn=worker_init_fix if fix_seed else worker_init_rand,
        )

    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.L1Loss()
    loss_fn = HammingLoss()
    # loss_fn = CycleLoss()
    
    #   
    set_deterministic()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=30)
    

    model.train()
    num_instances, num_epochs, eval_freq = 0, 100, 10
    loss = 0.0
    for i in range(num_epochs):
        first_iter, final_iter = 0, len(ts_dataloader)
        progress_bar = tqdm(range(first_iter, final_iter), desc="Dataloader")
        iteration, update_freq = 1, 10

        for inputs in ts_dataloader:

            image_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            edges_list = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]
            
            images = torch.stack(image_list, dim=0)
            pts_gt = torch.stack(points_gt_list, dim=0)
            n_pts_gt = torch.stack(n_points_gt_list, dim=0)
            n_pts_per_graph = n_pts_gt[0]

            if (n_pts_per_graph[0] <=2):    #   Skip small graphs
                continue

            predicted_matching = model(image_list, points_gt_list, edges_list, n_points_gt_list, perm_mat_list)    
                
            if predicted_matching == None:
                continue
            loss = 0
            for gt_match, pred_match in zip(perm_mat_list, predicted_matching):
                loss += loss_fn(pred_match.flatten(),gt_match.flatten())

            loss.backward()
            optimizer.step()
            # scheduler.step()
            iteration += 1

            if (iteration % update_freq == 0):
                progress_bar.update(update_freq)

            if iteration % 100 == 0:
                break

        accs, f1_scores = do_evaluation(model, ds, ts_dataloader)


