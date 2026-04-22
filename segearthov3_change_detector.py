import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage import measure
from skimage.filters import threshold_otsu

from mmseg.models.segmentors import BaseSegmentor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_class_indices(path: str):
    """
    Parse class names and their corresponding indices from a given text file.
    """
    with open(path, 'r') as f:
        name_sets = f.readlines()
    
    class_names, class_indices = [], []
    for idx, line in enumerate(name_sets):
        names_i = [i.strip() for i in line.split(',')]
        class_names.extend(names_i)
        class_indices.extend([idx] * len(names_i))
    
    return class_names, class_indices


@MODELS.register_module()
class SegEarthOV3CDSeg(BaseSegmentor):
    def __init__(self, 
                 classname_path: str,
                 device=torch.device('cuda'),
                 prob_thd: float = 0.0,
                 bg_idx: int = 0,
                 slide_stride: int = 0,
                 slide_crop: int = 0,
                 confidence_threshold: float = 0.5,
                 use_sem_seg: bool = True,
                 use_presence_score: bool = True,
                 use_transformer_decoder: bool = True,
                 **kwargs):
        super().__init__()
        
        self.device = device
        
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path='weights/sam3/sam3.pt', 
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        
        self.query_words, self.query_idx = load_class_indices(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        
        # Inference feature flags
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder
       
        # Hybrid Method Parameters
        self.instance_iou_threshold = kwargs.get('instance_iou_threshold', 0.3)  
        self.t12_min_instance_area = kwargs.get('t12_min_instance_area', 20)  

    def _extract_fpn_features(self, image: Image.Image):
        """Extract FPN features using the SAM3 backbone."""
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            if "backbone_out" not in inference_state:
                raise ValueError("backbone_out not found in inference_state")
            if "backbone_fpn" not in inference_state["backbone_out"]:
                raise ValueError("backbone_fpn not found in backbone_out")
                
            fpn_features = inference_state["backbone_out"]["backbone_fpn"]
            original_size = (image.size[1], image.size[0])
            
            return fpn_features, original_size

    def _compute_fpn_similarity_map(self, fpn_features_t1, fpn_features_t2, target_size):
        """Compute cosine similarity between two sets of FPN features."""
        def process_fpn_features(fpn_features, target_size):
            upsampled_features = []
            for feat in fpn_features[-1:]:
                B, C, H, W = feat.shape
                if C > 320:
                    feat_part1 = feat[:, :320, :, :]
                    feat_part2 = feat[:, 320:, :, :]
                    feat_part1_upsampled = F.interpolate(feat_part1, size=target_size, mode='bilinear', align_corners=False)
                    feat_part2_upsampled = F.interpolate(feat_part2, size=target_size, mode='bilinear', align_corners=False)
                    feat_upsampled = torch.cat([feat_part1_upsampled, feat_part2_upsampled], dim=1)
                else:
                    feat_upsampled = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                upsampled_features.append(feat_upsampled)
            return torch.cat(upsampled_features, dim=1)
        
        feat_t1 = process_fpn_features(fpn_features_t1, target_size)  
        feat_t2 = process_fpn_features(fpn_features_t2, target_size)

        cosine_sim = F.cosine_similarity(feat_t1, feat_t2, dim=1)
        
        similarity_map = cosine_sim * 0.5 + 0.5
        return similarity_map.squeeze(1)

    def _inference_single_view(self, image: Image.Image):
        """Inference on a single PIL image or crop patch, returning unified logits."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                        semantic_logits = F.interpolate(
                            semantic_logits, 
                            size=(h, w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze()
                    
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                
                if self.use_presence_score:
                    seg_logits[query_idx] = seg_logits[query_idx] * inference_state["presence_score"]
                
        return seg_logits

    def slide_inference(self, image: Image.Image, stride, crop_size):
        """Inference by sliding-window with overlap using PIL cropping."""
        w_img, h_img = image.size
        
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(crop_size, int): crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        
        preds = torch.zeros((self.num_queries, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, h_img, w_img), device=self.device)
        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                crop_img = image.crop((x1, y1, x2, y2))
                crop_seg_logit = self._inference_single_view(crop_img)
                
                preds[:, y1:y2, x1:x2] += crop_seg_logit
                count_mat[:, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."
        return preds / count_mat

    def _get_discrete_pred(self, logits):
        """Convert continuous logits to discrete class predictions."""
        if self.num_cls != self.num_queries:
            logits_tmp = logits.unsqueeze(0)
            cls_index = nn.functional.one_hot(self.query_idx).T.view(self.num_cls, len(self.query_idx), 1, 1)
            pred = (logits_tmp * cls_index).max(1)[0].argmax(0)
        else:
            pred = torch.argmax(logits, dim=0)
        
        max_vals = logits.max(0)[0]
        pred[max_vals < self.prob_thd] = self.bg_idx
            
        return pred

    def predict(self, inputs, data_samples):
        """Main execution method processing the batch."""
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:], pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            ori_shape = meta['ori_shape']

            # ==============================================================
            # 1. Process Time 1 Image
            # ==============================================================
            img_path_t1 = meta.get('img1_path')
            img_t1 = Image.open(img_path_t1).convert('RGB')
            if 'img_path' not in meta and img_path_t1 is not None:
                data_samples[i].set_metainfo(dict(img_path=img_path_t1))

            if self.slide_crop > 0 and (self.slide_crop < img_t1.size[0] or self.slide_crop < img_t1.size[1]):
                logits_t1 = self.slide_inference(img_t1, self.slide_stride, self.slide_crop)
            else:
                logits_t1 = self._inference_single_view(img_t1)

            if logits_t1.shape[-2:] != ori_shape:
                logits_t1 = F.interpolate(logits_t1.unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=False).squeeze(0)

            # ==============================================================
            # 2. Process Time 2 Image
            # ==============================================================
            img_path_t2 = meta.get('img2_path')
            img_t2 = Image.open(img_path_t2).convert('RGB')

            if self.slide_crop > 0 and (self.slide_crop < img_t2.size[0] or self.slide_crop < img_t2.size[1]):
                logits_t2 = self.slide_inference(img_t2, self.slide_stride, self.slide_crop)
            else:
                logits_t2 = self._inference_single_view(img_t2)

            if logits_t2.shape[-2:] != ori_shape:
                logits_t2 = F.interpolate(logits_t2.unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=False).squeeze(0)

            # ==============================================================
            # 3. Change Detection Logic
            # ==============================================================
            
            # 3.1 Instance Level Prediction Extraction & Matching
            pred_inst_t1 = self._get_discrete_pred(logits_t1)
            pred_inst_t2 = self._get_discrete_pred(logits_t2)
            
            change_pred_inst = self._detect_instance_changes(pred_inst_t1, pred_inst_t2)

            # 3.2 Extract category-agnostic features and compute physical change map
            fpn_t1, _ = self._extract_fpn_features(img_t1)
            fpn_t2, _ = self._extract_fpn_features(img_t2)
            
            sim_map = self._compute_fpn_similarity_map(fpn_t1, fpn_t2, ori_shape).squeeze(0)
            feature_change_map = 1.0 - sim_map  # 0 means no change, 1 means huge structural change

            change_pred_sem = torch.full(ori_shape, self.bg_idx, dtype=torch.long, device=self.device)

            # # 3.3 Construct Joint Energy Map
            for cls_id in range(self.num_cls):
                if cls_id == self.bg_idx: 
                    continue
                
                prompt_indices = (self.query_idx == cls_id).nonzero(as_tuple=True)[0]
                if len(prompt_indices) == 0:
                    continue
                
                delta_prompts = torch.abs(logits_t1[prompt_indices] - logits_t2[prompt_indices])
                delta_c_max = delta_prompts.max(dim=0)[0]  
                
                max_conf_prompts = torch.max(logits_t1[prompt_indices], logits_t2[prompt_indices])
                max_conf_c = max_conf_prompts.max(dim=0)[0]  
                
                energy_map = max_conf_c * delta_c_max * feature_change_map
                energy_np = energy_map.cpu().numpy()
                if np.ptp(energy_np) > 1e-4:
                    otsu_thd = threshold_otsu(energy_np)
                else:
                    otsu_thd = 0.0
                
                valid_semantic_mask = energy_map > max(otsu_thd, 0.01)
                change_pred_sem[valid_semantic_mask] = cls_id
            
            # 3.4 Mask Merge (Intersection Strategy)
            # Strictly keep areas predicted as changed by both instance and semantic methods
            agreed_mask = (change_pred_inst != self.bg_idx) & (change_pred_sem != self.bg_idx)
            change_pred = torch.full(ori_shape, self.bg_idx, dtype=torch.long, device=self.device)
            change_pred[agreed_mask] = change_pred_inst[agreed_mask]
            
            pred_sem_seg_data = change_pred.unsqueeze(0)  

            data_samples[i].set_data({
                'seg1_logits': PixelData(**{'data': logits_t1}), 
                'seg2_logits': PixelData(**{'data': logits_t2}), 
                'pred_sem_seg': PixelData(**{'data': pred_sem_seg_data})
            })
            
        return data_samples

    def _detect_instance_changes(self, mask_t1, mask_t2, inst_dilation_radius=2):
        """
        Fast and robust vectorized instance-level change detection.
        Solves registration drift and 1-to-N fragmentation issues rapidly.
        """
        # 1. Extract foreground boolean masks
        mask1 = (mask_t1 > 0).float().unsqueeze(0).unsqueeze(0)
        mask2 = (mask_t2 > 0).float().unsqueeze(0).unsqueeze(0)

        # 2. Registration Error Buffer (Dilation tolerance)
        if inst_dilation_radius > 0:
            kernel_size = inst_dilation_radius * 2 + 1
            mask1_dilated = F.max_pool2d(mask1, kernel_size, stride=1, padding=inst_dilation_radius)
            mask2_dilated = F.max_pool2d(mask2, kernel_size, stride=1, padding=inst_dilation_radius)
        else:
            mask1_dilated = mask1
            mask2_dilated = mask2
            
        mask1 = mask1.squeeze().cpu().numpy() > 0
        mask2 = mask2.squeeze().cpu().numpy() > 0
        mask1_dilated = mask1_dilated.squeeze().cpu().numpy() > 0
        mask2_dilated = mask2_dilated.squeeze().cpu().numpy() > 0

        # 3. Helper function for unidirectional change detection
        def get_unidirectional_changes(source_mask, target_mask_dilated):
            """Calculate changes from source to target direction."""
            lbl = measure.label(source_mask, connectivity=2)
            
            area = np.bincount(lbl.ravel())
            cover = np.bincount(lbl.ravel(), weights=target_mask_dilated.ravel().astype(np.float64))
            ratio = cover / np.maximum(area, 1)

            changed_ids = np.where(
                (ratio < self.instance_iou_threshold) & (area >= self.t12_min_instance_area)
            )[0]
            changed_ids = changed_ids[changed_ids != 0]  

            return np.isin(lbl, changed_ids)

        # 4. Compute bidirectional changes
        change_t1 = get_unidirectional_changes(mask1, mask2_dilated)
        change_t2 = get_unidirectional_changes(mask2, mask1_dilated)

        # 5. Merge changes
        change_np = change_t1 | change_t2

        return torch.from_numpy(change_np).long().to(mask_t1.device)
    
    # -------------------------------------------------------------
    # MMSegmentation Required Abstract Methods
    # -------------------------------------------------------------
    def _forward(self, data_samples): pass
    def inference(self, img, batch_img_metas): pass
    def encode_decode(self, inputs, batch_img_metas): pass
    def extract_feat(self, inputs): pass
    def loss(self, inputs, data_samples): pass