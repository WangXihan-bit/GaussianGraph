import os
import re
import sys
import json
import random
import argparse
import cv2
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass, field
from typing import Tuple, Type
import open3d as o3d
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import torchvision
from torch import nn
from loguru import logger
try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from submodules.segment_anything.sam2.build_sam import build_sam2
from submodules.segment_anything.sam2.automatic_mask_generator_2 import SAM2AutomaticMaskGenerator
from submodules.segment_anything.sam2.sam2_image_predictor import SAM2ImagePredictor
from submodules.groundingdino.groundingdino.util.inference import Model
from submodules.llava.utils import disable_torch_init
from submodules.llava.model.builder import load_pretrained_model
from submodules.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from submodules.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from submodules.llava.conversation import conv_templates

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "/home/wangxihan/LangSplat/LangSplat-main/submodules/open_clip/open_clip_pytorch_model.bin"
    clip_n_dims: int = 512
   
class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type, 
            self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to(args.device)
        self.clip_n_dims = self.config.clip_n_dims

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
    def encode_texts(self, class_ids, classes):
        with torch.no_grad():
            tokenized_texts = torch.cat([self.tokenizer(classes[class_id]) for class_id in class_ids]).to(args.device)
            text_feats = self.model.encode_text(tokenized_texts)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        return text_feats

class LLaVaChat():
    # Model Constants
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"

    def __init__(self, model_path):
        disable_torch_init()

        self.model_name = get_model_name_from_path(model_path)  
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name, device="cuda")

        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

    def preprocess_image(self, images):
        x = process_images(
            images,
            self.image_processor,
            self.model.config)

        return x.to(self.model.device, dtype=torch.float16)

    def __call__(self, query, image_features, image_sizes):
        # Given this query, and the image_featurese, prompt LLaVA with the query,
        # using the image_features as context.

        conv = conv_templates[self.conv_mode].copy()

        if self.model.config.mm_use_im_start_end:
            inp = LLaVaChat.DEFAULT_IM_START_TOKEN +\
                  LLaVaChat.DEFAULT_IMAGE_TOKEN +\
                  LLaVaChat.DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            inp = LLaVaChat.DEFAULT_IMAGE_TOKEN + '\n' + query
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, LLaVaChat.IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.temperature = 0
        self.max_new_tokens = 512
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_features,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.decode(output_ids[0]).strip()
        return outputs

def describe_LLAVA(mask_id, image, chat:LLaVaChat, mode):

    ### caption
    image_sizes = [image.size]
    image_tensor = chat.preprocess_image([image]).to("cuda", dtype=torch.float16)
    template = {}

    if mode == "category":
        query_base = """Describe visible object categories in front of you."""
        query_tail = """
        Only provide the category names, no descriptions needed
        If there are multiple objects belonging to the same category, 
        you only need to output the category name once to avoid duplication.
        Examples:
        'Chair', 'Table', 'Chalkboard'
        'Book', 'Bookshelf', 'Window'
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
        template["categories"] = re.sub(r'\s+', ' ', text.replace("<s>", "").replace("</s>", "").replace("-", "").strip())

    if mode == "captions":
        query_base = """Describe visible object in front of you, 
        paying close attention to its spatial dimensions and visual attributes."""
        
        query_tail = """
        The object is one we usually see in indoor scenes. 
        It signature must be short and sparse, describe appearance, geometry, material. Don't describe background.
        Fit you description in four or five words.
        Examples: 
        a closed wooden door with a glass panel;
        a pillow with a floral pattern;
        a wooden table;
        a gray wall.
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
        template["id"] = mask_id
        template["description"] = text.replace("<s>", "").replace("</s>", "").strip()

    elif mode == "relationships":
        query_base = """There are two objects selected by the red and green rectangular boxes, 
        paying close attention to the positional relationship between two selected objects."""
        query_tail = """
        You are a vision-language model capable of analyzing spatial relationships between objects in an image. In the given image, there are two boxed objects:
        - The object selected by the red box is [Object A].
        - The object selected by the blue box is [Object B].

        Based on the image content, analyze and describe the spatial relationship between these two objects. The spatial relationships may include the following types:
        - Above/below (e.g., "Object A is above Object B")
        - Left/right (e.g., "Object A is to the left of Object B")
        - In front/behind (e.g., "Object A is in front of Object B")
        - Containing (e.g., "Object A is contained within Object B")
        - Adjacent (e.g., "Object A is next to Object B")

        Please provide the output in the following format:
        1. Object A and Object B names.
        2. The spatial relationship between Object A and Object B.
        3. A detailed description of the relationship (optional).

        Example output:
        1. Object A: Chair; Object B: Table.
        2. Spatial relationship: The chair is to the left of the table.
        3. Detailed description: The chair is placed close to the table, facing toward it.

        Generate the results based on the image content.
        """
        query = query_base + "\n" + query_tail
        text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
        template["id_pair"] = mask_id
        template["relationship"] = text.replace("<s>", "").replace("</s>", "").strip()

    return template

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        if not masks_lvl:
            masks_new += ([],)  # 或者其他适当的处理
            continue
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def compute_iou_matrix(generated_masks, seg_map, unique_mask_indices):
    """
    计算所有生成的掩码与seg_map中所有掩码的IoU
    :param generated_masks: (num_masks, H, W)，生成的掩码数组
    :param seg_map: (H, W), 全图的seg_map
    :param unique_mask_indices: seg_map中唯一的掩码索引
    :return: (num_masks, num_seg_masks) 的IoU矩阵
    """
    num_seg_masks = len(unique_mask_indices)
    generated_masks = generated_masks.astype(np.bool_)
    # 初始化一个空的IoU矩阵
    iou_matrix = np.zeros((1, num_seg_masks))
    
    # 逐个计算IoU
    for i, mask_index in enumerate(unique_mask_indices):
        if mask_index == -1:  # 跳过背景
            continue
        
        # 获取seg_map中当前掩码的区域
        seg_mask = (seg_map == mask_index)  # (H, W)
        
        # 计算交集和并集
        intersection = np.sum(generated_masks & seg_mask)  # (num_masks, H, W) 与 (H, W) 计算交集
        union = np.sum(generated_masks | seg_mask)  # (num_masks, H, W) 与 (H, W) 计算并集
        
        # 计算IoU
        iou_matrix[:, i] = intersection / (union + 1e-6)  # 防止除以零，1e-6为小常数

    return iou_matrix

def get_bbox_img(box, image):
    image = image.copy()
    x_min, y_min, x_max, y_max = map(int, box)
    # 从图像中截取框内区域
    seg_img = image[y_min:y_max, x_min:x_max]
    return seg_img

def sam_predictor(seg_map, image, detections):

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor_sam.set_image(image=image)
        seg_img_list = []
        classes_list = []
        mask_indices = {}

        # 获取所有模式下唯一的掩码索引（减少冗余计算）
        unique_mask_indices_cache = {}
        for mode in ['default', 's', 'm', 'l']:
            unique_mask_indices_cache[mode] = np.unique(seg_map[mode])

        for i, box in enumerate(detections.xyxy):
            category_id = detections.class_id[i]
            classes_list.append(category_id)
            masks, scores, logits = predictor_sam.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            generated_mask = masks[index]
            
            mode_mask_indices = {}
            for mode, unique_mask_indices in unique_mask_indices_cache.items():
                
                # 计算IoU矩阵
                iou_matrix = compute_iou_matrix(generated_mask[None, :, :], seg_map[mode], unique_mask_indices)
                best_mask_index = unique_mask_indices[np.argmax(iou_matrix)]
                mode_mask_indices[mode] = best_mask_index

            mask_indices[i] = mode_mask_indices

            seg_img = get_bbox_img(box, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)
        
        if len(classes_list) > 0:
            catogories = torch.from_numpy(np.stack(classes_list, axis=0))
            seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
            seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to(args.device)

    return seg_imgs, catogories, mask_indices

def sam_encoder(image):
    
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    #masks_default, masks_s, masks_m, masks_l = masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.7, score_thr=0.6, inner_thr=0.5)
   
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
      
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map
    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def is_overlapping(box1, box2):
    """
    Check if two bounding boxes overlap.
    
    Args:
        box1 (list or array): Coordinates of the first box [x1, y1, x2, y2].
        box2 (list or array): Coordinates of the second box [x1, y1, x2, y2].
    
    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check if there is no overlap
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    return True

def crop_and_blackout(image, bbox1, bbox2, padding):
    """
    从图像中截取指定索引的两个矩形框区域，并将其余部分设为黑色。
    
    参数：
        image: 输入图像 (H, W, C)
        detections: 检测框列表，每个元素是一个 [x1, y1, x2, y2]
        idx1: 第一个矩形框的索引
        idx2: 第二个矩形框的索引
        
    返回：
        cropped_image: 包含两个矩形框的图像，其他区域为黑色
    """
    height, width = image.shape[:2]
    # 复制图像，初始化为黑色图像
    cropped_image = np.zeros_like(image)
    
    # 获取第一个矩形框的坐标
    x1, y1, x2, y2 = map(int, bbox1)
    # 扩充裁剪区域，确保不会超出图像边界
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    # 将第一个矩形框区域复制到黑色图像中
    cropped_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    # 获取第二个矩形框的坐标
    x1, y1, x2, y2 = map(int, bbox2)
    # 扩充裁剪区域，确保不会超出图像边界
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    # 将第二个矩形框区域复制到黑色图像中
    cropped_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    
    return cropped_image

def draw_bounding_boxes(image, bbox1, bbox2, color1=(0, 255, 0), color2=(0, 0, 255), thickness=2):
    """
    在图像上绘制两个矩形框。
    
    参数：
        image: 输入图像 (H, W, C)
        bbox1: 第一个矩形框的坐标 [x1, y1, x2, y2]
        bbox2: 第二个矩形框的坐标 [x1, y1, x2, y2]
        color1: 第一个矩形框的颜色 (B, G, R)
        color2: 第二个矩形框的颜色 (B, G, R)
        thickness: 矩形框的线条粗细
    """
    x1_min, y1_min, x1_max, y1_max = map(int, bbox1)
    x2_min, y2_min, x2_max, y2_max = map(int, bbox2)
    
    # 绘制第一个矩形框
    cv2.rectangle(image, (x1_min, y1_min), (x1_max, y1_max), color1, thickness)
    
    # 绘制第二个矩形框
    cv2.rectangle(image, (x2_min, y2_min), (x2_max, y2_max), color2, thickness)

def graph_construct(image_path, sam_predictor, sam_encoder, llava_chat, classes_set):
    
    image_pil = Image.open(image_path).convert("RGB")
    image = cv2.imread(image_path)
    resolution = (800, 800)  
    image = cv2.resize(image, resolution)
    image_pil = image_pil.resize((resolution[1], resolution[0]), Image.ANTIALIAS)

    seg_images, seg_map = sam_encoder(np.array(image_pil))

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = clip_model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()

    graph_dict = {}
    print(image_path, '******************')
    with torch.no_grad():
        classes_info = describe_LLAVA(mask_id=None, image=image_pil, chat=llava_chat, mode='category')
   
        classes = list(set(classes_info['categories'].split('\n')))
        print(classes, 'class')
        classes_set.update(classes)

        # grounding_dino detector
        if len(classes) > 0:
            classes = classes
        else:
            assert len(classes) == 0, "Error: No target detected in the image!"

        graph_dict['classes'] = classes

        detections = grounding_dino_model.predict_with_classes(
            image=image, # This function expects a BGR image...
            classes=classes,
            box_threshold=0.2,
            text_threshold=0.2,
        )
        
        if len(detections.class_id) > 0:
            ### Non-maximum suppression ###
            # print(f"Before NMS: {len(detections.xyxy)} boxes")
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                args.nms_threshold
            ).numpy().tolist()
            # print(f"After NMS: {len(detections.xyxy)} boxes")

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            
            # Somehow some detections will have class_id=-1, remove them
            valid_idx = detections.class_id != -1
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]

        else:
            detections = grounding_dino_model.predict_with_classes(
            image=image, # This function expects a BGR image...
            classes=classes,
            box_threshold=0.1,
            text_threshold=0.1,
            )

            assert len(detections.class_id) == 0, "Error: No target detected in the image!"

        # sam segmentation
        seg_bbox, categories, match_indices = sam_predictor(seg_map, image, detections)

        # clip
        tiles = seg_bbox.to(args.device)

        # captions of foreground objects
        descriptions = []
        for idx, fore_box in enumerate(detections.xyxy):
            cropped_image = np.zeros_like(image_pil)

            # 获取矩形框的坐标
            x1, y1, x2, y2 = map(int, fore_box)
            cropped_image[y1:y2, x1:x2] = np.array(image_pil)[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped_image)
            
            match_idx = {}
            for mode in ['default', 's', 'm', 'l']:
                match_idx[mode] = int(match_indices[idx][mode])

            description = describe_LLAVA(mask_id=match_idx, image=cropped_image, chat=llava_chat, mode='captions')
            descriptions.append(description)

        graph_dict['captions'] = descriptions

        image_embed = clip_model.encode_image(tiles)
        image_embed /= image_embed.norm(dim=-1, keepdim=True)
        # text_embed = clip_model.encode_texts(categories, classes)

        # generate relation
        relations = []
        for idx_i, bbox_i in enumerate(detections.xyxy):
            for idx_j, bbox_j in enumerate(detections.xyxy[idx_i + 1:], start=idx_i + 1):
                if idx_i == idx_j:
                    continue
                torch.cuda.empty_cache()
                # 计算特征相似度
                similarity = torch.cosine_similarity(image_embed[idx_i].unsqueeze(0), 
                                                    image_embed[idx_j].unsqueeze(0), dim=1).item()
                    
                inter = is_overlapping(detections.xyxy[idx_i], detections.xyxy[idx_j])

                if similarity > 0.5 or inter:
                    match_idx_i = {}
                    match_idx_j = {}
                    for mode in ['default', 's', 'm', 'l']:
                        match_idx_i[mode] = int(match_indices[idx_i][mode])
                        match_idx_j[mode] = int(match_indices[idx_j][mode])
                    image_copy = image.copy()
                    draw_bounding_boxes(image_copy, detections.xyxy[idx_i], detections.xyxy[idx_j])
                    boxed_image = Image.fromarray(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
                    output_path = os.path.join(args.output_dir, f"object_i{idx_i}_j{idx_j}.png")
                    boxed_image.save(output_path)
                    relation_info = describe_LLAVA(mask_id=(match_idx_i, match_idx_j), image=boxed_image, chat=llava_chat, mode='relationships')
                    relations.append(relation_info)
    
        graph_dict['relations'] = relations
    
    return clip_embeds, seg_map, graph_dict

def create(args, img_folder, save_folder):
    data_list = os.listdir(img_folder)
    data_list.sort()
    assert len(data_list) is not None, "image_list must be provided to generate features"
    timer = 0
    embed_size=512
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(data_list), 100, embed_size))
    seg_maps = torch.zeros((len(data_list), 4, 800, 800)) 
    llava_chat = LLaVaChat(args.llava_ckpt)
    classes_set = set()
    mask_generator.predictor.model

    for i, data_path in tqdm(enumerate(data_list), desc="Embedding images", leave=False):
        timer += 1
        torch.cuda.empty_cache()
        image_path = os.path.join(img_folder, data_path)

        img_embed, seg_map, graph_dict = graph_construct(image_path, sam_predictor, sam_encoder, llava_chat, classes_set)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)

        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(data_list), pad, embed_size))
            ], dim=1)
        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
      
        img_embeds[i, :total_length] = img_embed
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map
      
        # 保存每个图像的 img_embed, seg_map和rel_info
        save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(image_path))[0])

        # 确保 seg_map 的最大值与长度一致
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i],
            'graph': graph_dict
        }

        sava_numpy(save_path, curr)
    
def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    save_path_r = save_path + '_r.json'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())
    with open(save_path_r, 'w') as f:
        json.dump(data['graph'], f)

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="sam2.1_hiera_l.yaml")
    parser.add_argument('--sam_ckpt', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--gsa_config', type=str, default="groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--gsa_ckpt', type=str, default="groundingdino/groundingdino_swint_ogc.pth")
    parser.add_argument('--llava_ckpt', type=str, default="llava/llava-next/llava_1.6")
    parser.add_argument("--box_threshold", type=float, default=0.2)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    parser.add_argument("--nms_threshold", type=float, default=0.2)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path

    # 判断路径是否存在
    if os.path.exists(os.path.join(dataset_path, 'color')):
        img_folder = os.path.join(dataset_path, 'color')
    elif os.path.exists(os.path.join(dataset_path, 'images')):
        img_folder = os.path.join(dataset_path, 'images')
    else:
        raise ValueError('Image folder not found')

    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    grounding_dino_model = Model(model_config_path=args.gsa_config, model_checkpoint_path=args.gsa_ckpt, device=args.device)
    sam = build_sam2(args.config, args.sam_ckpt, args.device, apply_postprocessing=False)
    predictor_sam = SAM2ImagePredictor(sam_model=sam)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam)
    WARNED = False

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    create(args, img_folder, save_folder)
