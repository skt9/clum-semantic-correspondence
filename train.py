#   Taken from Deep Blackbox Graph Matching: https://github.com/martius-lab/blackbox-deep-graph-matching.
import torch
import torch.optim as optim
import time
from pathlib import Path

from data.data_loader import GraphMatchingDataset, get_dataloader

from CLUM.model import MainNetwork

from utils.cfg import cfg
from utils.evaluation_metric import matching_accuracy_from_lists, f1_score
import json
import os
from utils.utils import update_params_from_cmdline

lr_schedules = {
    "long_halving": (10, (2, 4, 6, 8, 10), 0.5),
    "short_halving": (2, (1,), 0.5),
    "long_nodrop": (10, (10,), 1.0),
    "minirun": (1, (10,), 1.0),
}

class CycleLoss(torch.nn.Module):
    def forward(self, predicted, target):

        # Check if the dimensions are the same
        if (predicted.size() != target.size()):
            raise ValueError("Both matrices must have the same dimensions.")

        # Count the number of differing locations using element-wise comparison
        count = torch.sum(predicted != target)

        return count


def train_loop(model, dataloader, optimizer, num_epochs):

    start_epoch = 0.
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        model.train()  # Set model to training mode

        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        iter_num = 0


        for inputs in dataloader['train']:
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            edges_list = [_.to("cuda") for _ in inputs["edges"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            optimizer.zero_grad()
            iter_num = iter_num + 1
            s_pred_list = model(data_list, points_gt_list, edges_list, perm_mat_list)

            loss = sum([criterion(s_pred, perm_mat) for s_pred, perm_mat in zip(s_pred_list, perm_mat_list)])
            loss /= len(s_pred_list)

            # backward + optimize
            loss.backward()
            optimizer.step()



if __name__ == "__main__":

    
    # print(cfg)
    print(cfg['TRAIN'])
    cfg = update_params_from_cmdline(default_params=cfg)

    print(cfg)
    
    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GraphMatchingDataset(cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256)) for x in ("train", "test")
    }


    model = MainNetwork()
    model = model.cuda()

    criterion = CycleLoss()

    #   Optimizer
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == "test")) for x in ("train", "test")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    backbone_params = list(model.node_layers.parameters()) + list(model.edge_layers.parameters())
    backbone_params += list(model.final_layers.parameters())

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=cfg.TRAIN.LR * 0.01),
        dict(params=new_params, lr=cfg.TRAIN.LR),
    ]
    optimizer = optim.Adam(opt_params)

    train_loop(model, dataloader, optimizer, cfg.TRAIN.NUM_EPOCHS)