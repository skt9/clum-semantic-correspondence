import glob
import json
import os
import pickle
import random
import torch
import numpy as np
from PIL import Image
from collections import namedtuple
from glob import glob
from utils.config import cfg
from torch.utils.data import Dataset
from utils.build_graphs import build_graphs
from torch_geometric.data import Data, Batch
import torchvision.models as tvm
import torchvision.transforms as transforms

cache_path = cfg.CACHE_PATH
pair_ann_path = cfg.SPair.ROOT_DIR + "/PairAnnotation"
layout_path = cfg.SPair.ROOT_DIR + "/Layout"
image_path = cfg.SPair.ROOT_DIR + "/JPEGImages"
dataset_size = cfg.SPair.size

sets_translation_dict = dict(train="trn", test="test")
difficulty_params_dict = dict(
    trn=cfg.TRAIN.difficulty_params, val=cfg.EVAL.difficulty_params, test=cfg.EVAL.difficulty_params
)
TripletInfo = namedtuple('TripletInfo', ['im1','im2','im3','kpts1', 'kpts2', 'kpts3'])
Assignment = namedtuple('Assignment', 'left right cost')
Edge = namedtuple('Edge', 'assignment1 assignment2 cost')

class SPair71K(Dataset):
    
    def __init__(self,  ds_path: str, dataset_size: str = "small",\
                  mode: str="train", num_graphs_per_instance: int=3,
                    num_iters: int = 100000):
        """
            SPair71K dataset
        """
        self.main_path = ds_path
        self.pair_anno_path = os.path.join(ds_path,"PairAnnotation")
        self.triplet_anno_path = os.path.join(ds_path,"TripletAnnotation")
        self.layout_path = os.path.join(ds_path,"Layout")
        self.image_path = os.path.join(ds_path,"JPEGImages")
        self.set = mode
        self.classes = cfg.SPair.CLASSES
        self.combine_classes = False
        self.dataset_size = dataset_size
        self.anno_files = open(os.path.join(self.layout_path, self.dataset_size, self.set + ".txt"), "r").read().split("\n")
        mode_str=os.path.join(self.layout_path, self.dataset_size, self.set + ".txt")
        self.anno_files = self.anno_files[: len(self.anno_files) - 1]
        self.difficulty_params = difficulty_params_dict[self.set]
        self.anno_files_filtered, self.anno_files_filtered_cls_dict, self.classes = self.filter_annotations(
            self.anno_files, self.difficulty_params
        )
        self.triplet_files, self.triplets_by_cls_dict = self.read_triplet_annotations()
        if num_graphs_per_instance == 2:
            self.total_size = len(self.anno_files_filtered)
        else:
            self.total_size = len(self.triplet_files)
            
        self.size_by_cls = {cls: len(ann_list) for cls, ann_list in self.anno_files_filtered_cls_dict.items()}

        self.num_graphs_in_matching_instance = num_graphs_per_instance
        self.sampling_strategy = "intersection"
        self.cls = None
        self.num_iters = num_iters
        self.true_epochs = num_iters
        self.obj_resize = (256,256)
        
    def filter_annotations(self, ann_files, difficulty_params):
        if len(difficulty_params) > 0:
            basepath = os.path.join(self.pair_ann_path, "pickled", self.sets)
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            difficulty_paramas_str = self.diff_dict_to_str(difficulty_params)
            try:
                filepath = os.path.join(basepath, difficulty_paramas_str + ".pickle")
                ann_files_filtered = pickle.load(open(filepath, "rb"))
                print(
                    f"Found filtered annotations for difficulty parameters {difficulty_params} and {self.sets}-set at {filepath}"
                )
            except (OSError, IOError) as e:
                print(
                    f"No pickled annotations found for difficulty parameters {difficulty_params} and {self.sets}-set. Filtering..."
                )
                ann_files_filtered_dict = {}

                for ann_file in ann_files:
                    with open(os.path.join(self.pair_ann_path, self.sets, ann_file + ".json")) as f:
                        annotation = json.load(f)
                    diff = {key: annotation[key] for key in self.difficulty_params.keys()}
                    diff_str = self.diff_dict_to_str(diff)
                    if diff_str in ann_files_filtered_dict:
                        ann_files_filtered_dict[diff_str].append(ann_file)
                    else:
                        ann_files_filtered_dict[diff_str] = [ann_file]
                total_l = 0
                for diff_str, file_list in ann_files_filtered_dict.items():
                    total_l += len(file_list)
                    filepath = os.path.join(basepath, diff_str + ".pickle")
                    pickle.dump(file_list, open(filepath, "wb"))
                assert total_l == len(ann_files)
                print(f"Done filtering. Saved filtered annotations to {basepath}.")
                ann_files_filtered = ann_files_filtered_dict[difficulty_paramas_str]
        else:
            print(f"No difficulty parameters for {self.set}-set. Using all available data.")
            ann_files_filtered = ann_files

        ann_files_filtered_cls_dict = {
            cls: list(filter(lambda x: cls in x, ann_files_filtered)) for cls in self.classes
        }
        class_len = {cls: len(ann_list) for cls, ann_list in ann_files_filtered_cls_dict.items()}
        print(f"Number of annotation pairs matching the difficulty params in {self.set}-set: {class_len}")
        if self.combine_classes:
            cls_name = "combined"
            ann_files_filtered_cls_dict = {cls_name: ann_files_filtered}
            filtered_classes = [cls_name]
            print(f"Combining {self.set}-set classes. Total of {len(ann_files_filtered)} image pairs used.")
        else:
            filtered_classes = []
            for cls, ann_f in ann_files_filtered_cls_dict.items():
                if len(ann_f) > 0:
                    filtered_classes.append(cls)
                else:
                    print(f"Excluding class {cls} from {self.set}-set.")
        return ann_files_filtered, ann_files_filtered_cls_dict, filtered_classes

    def set_num_graphs(self, num_graphs):
        self.num_graphs_in_matching_instance = num_graphs

    def read_triplet_annotations(self):
        triplet_files = sorted(glob(os.path.join(self.triplet_anno_path,"*.json")))
        triplets = {}
        for trip_file in triplet_files:
            with open(trip_file,"r") as f:
                data_dict = json.load(f)
                category = data_dict['category']
                if category not in triplets.keys():
                    triplets[category] = []
                main_file = os.path.split(trip_file)[1].split('.')[0]
                triplets[category].append(main_file)
        triplet_files = [os.path.split(trip)[1].split('.')[0] for trip in triplet_files]
        return triplet_files, triplets

    def diff_dict_to_str(self, diff):
        diff_str = ""
        keys = ["mirror", "viewpoint_variation", "scale_variation", "truncation", "occlusion"]
        for key in keys:
            if key in diff.keys():
                diff_str += key
                diff_str += str(diff[key])
        return diff_str

    def get_k_samples(self, idx, k, mode, cls=None, shuffle=True):
        """
        Randomly get a sample of k objects dataset
        :param idx: Index of datapoint to sample, None for random sampling
        :param k: number of datapoints in sample
        :param mode: sampling strategy
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (k samples of data, k \choose 2 groundtruth permutation matrices)
        """
        if k not in  [2,3]:
            raise NotImplementedError(
                f"No strategy implemented to sample {k} graphs from SPair dataset. So far only k = 2 or 3 is possible."
            )

        #   Pick a random class
        cls = self.classes[random.randrange(0, len(self.classes))]
        if k==2:
            #   Get the annotation files for the chosen class
            ann_files = self.anno_files_filtered_cls_dict[cls]
            # get pre-processed images
            assert len(ann_files) > 0
            ann_file = random.choice(ann_files) + ".json"
            with open(os.path.join(self.pair_anno_path, self.set, ann_file)) as f:
                annotation = json.load(f)

            category = annotation["category"]
            if cls is not None and not self.combine_classes:
                assert cls == category
            assert all(annotation[key] == value for key, value in self.difficulty_params.items())

            if mode == "intersection":
                assert len(annotation["src_kps"]) == len(annotation["trg_kps"])
                num_kps = len(annotation["src_kps"])
                perm_mat_init = np.eye(num_kps)
                anno_list, perm_list = [], []

                for st in ("src", "trg"):
                    if shuffle:
                        perm = np.random.permutation(np.arange(num_kps))
                    else:
                        perm = np.arange(num_kps)
                    kps = annotation[f"{st}_kps"]
                    img_path = os.path.join(self.image_path, category, annotation[f"{st}_imname"])
                    img, kps = self.rescale_im_and_kps(img_path, kps)
                    kps_permuted = [kps[i] for i in perm]
                    anno_dict = dict(image=img, keypoints=kps_permuted)
                    anno_list.append(anno_dict)
                    perm_list.append(perm)

                perm_mat = perm_mat_init[perm_list[0]][:, perm_list[1]]
                return anno_list, [perm_mat]
            else:
                raise NotImplementedError(f"Unknown sampling strategy {mode}")
            
        elif (k==3):
            ann_files = self.triplets_by_cls_dict[cls]
            # get pre-processed images
            assert len(ann_files) > 0
            ann_file = random.choice(ann_files) + ".json"
            ann_full_path = os.path.join(self.triplet_anno_path, ann_file)

            if not os.path.exists(ann_full_path):
                print(f"DOES NOT EXIST")

            with open(ann_full_path) as f:
                annotation = json.load(f)
                category = annotation["category"]

                kpts1, kpts2, kpts3 = annotation['im1_kpts'], annotation['im2_kpts'], annotation['im3_kpts']
                kps_list = [kpts1, kpts2, kpts3]
                num_kps = len(kpts1)
                perm_mat_init = np.eye(num_kps)
                anno_list, perm_list = [], []
                im_names = annotation['filename'].split('-')[1:]

                anno_list, perm_list = [], []
                for i,kps in enumerate(kps_list):
                    img_path = os.path.join(self.image_path, category, im_names[i] + ".jpg")
                    img, kps = self.rescale_im_and_kps(img_path,kps)
                    anno_dict = dict(image=img, keypoints=kps)
                    anno_list.append(anno_dict)

            return anno_list, [perm_mat_init]

    def rescale_im_and_kps(self, img_path, kps):

        with Image.open(str(img_path)) as img:
            w, h = img.size
            img = img.resize(self.obj_resize, resample=Image.BICUBIC)

        keypoint_list = []
        for kp in kps:
            x = kp[0] * self.obj_resize[0] / w
            y = kp[1] * self.obj_resize[1] / h
            keypoint_list.append(dict(x=x, y=y))

        return img, keypoint_list

    def len(self):
        return self.num_iters

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        """

        """
        # sampling_strategy = cfg.train_sampling if self.set == "trn" else cfg.eval_sampling
        # if self.num_graphs_in_matching_instance is None:
        #     raise ValueError("Num_graphs has to be set to an integer value.")
        sampling_strategy = self.sampling_strategy
        idx = idx if self.true_epochs else None
        anno_list, perm_mat_list = self.get_k_samples(idx, \
                        k=self.num_graphs_in_matching_instance, \
                            cls=self.cls, mode=sampling_strategy)
        for perm_mat in perm_mat_list:
            if (
                not perm_mat.size
                or (perm_mat.size < 2 * 2 and sampling_strategy == "intersection")
                and not self.true_epochs
            ):
                # 'and not self.true_epochs' because we assume all data is valid when sampling a true epoch
                next_idx = None if idx is None else idx + 1
                return self.__getitem__(next_idx)

        points_gt = [np.array([(kp["x"], kp["y"]) for kp in anno_dict["keypoints"]]) for anno_dict in anno_list]
        n_points_gt = [len(p_gt) for p_gt in points_gt]

        graph_list = []
        for p_gt, n_p_gt in zip(points_gt, n_points_gt):
            edge_indices, edge_features = build_graphs(p_gt, n_p_gt)

            # Add dummy node features so the __slices__ of them is saved when creating a batch
            pos = torch.tensor(p_gt).to(torch.float32) / 256.0
            assert (pos > -1e-5).all(), p_gt
            graph = Data(
                edge_attr=torch.tensor(edge_features).to(torch.float32),
                edge_index=torch.tensor(edge_indices, dtype=torch.long),
                x=pos,
                pos=pos,
            )
            graph.num_nodes = n_p_gt
            graph_list.append(graph)
        perm_mat_list = [torch.from_numpy(mat) for mat in perm_mat_list]
        ret_dict = {
            "Ps": [torch.Tensor(x) for x in points_gt],
            "ns": [torch.tensor(x) for x in n_points_gt],
            "gt_perm_mat": perm_mat_list,
            "edges": graph_list,
        }

        imgs = [anno["image"] for anno in anno_list]
        if imgs[0] is not None:
            # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.2)])
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)])
            # trans = transforms.Compose([transforms.ToTensor()])
            
            imgs = [trans(img) for img in imgs]
            ret_dict["images"] = imgs
        elif "feat" in anno_list[0]["keypoints"][0]:
            feat_list = [np.stack([kp["feat"] for kp in anno_dict["keypoints"]], axis=-1) for anno_dict in anno_list]
            ret_dict["features"] = [torch.Tensor(x) for x in feat_list]
        
        return ret_dict
