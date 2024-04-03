"""
Basic class agnostic counting model with backbone, refiner, matcher and counter.
"""
import torch
from torch import nn
from util.back import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


class CACModel(nn.Module):
    """ Class Agnostic Counting Model"""

    def __init__(self, backbone, backbone2, EPF_extractor, transformer, matcher, counter, hidden_dim):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            EPF_extractor: torch module of the feature extractor for patches. See epf_extractor.py
            repeat_times: Times to repeat each exemplar in the transformer decoder, i.e., the features of exemplar patches.
        """
        super().__init__()
        self.EPF_extractor = EPF_extractor

        self.matcher = matcher
        self.counter = counter
        self.backbone2 = backbone2
        self.backbone = backbone
        self.transformer = transformer
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(3, hidden_dim)

    def forward(self, samples: NestedTensor, patches: torch.Tensor, is_train: bool):
        """ The forward expects samples containing query images and corresponding exemplar patches.
            samples is a stack of query images, of shape [batch_size X 3 X H X W]    NestedTensor
            patches is a torch Tensor, of shape [batch_size x num_patches x 3 x 128 x 128]
            The size of patches are small than samples

            It returns a dict with the following elements:
               - "density_map": Shape= [batch_size x 1 X h_query X w_query]
               - "patch_feature": Features vectors for exemplars, not available during testing.
                                  They are used to compute similarity loss.
                                Shape= [exemplar_number x bs X hidden_dim]
               - "img_feature": Feature maps for query images, not available during testing.
                                Shape= [batch_size x hidden_dim X h_query X w_query]

        """
        # Stage 1: extract features for query images and exemplars
        scale_embedding, patches = patches['scale_embedding'], patches['patches']  # tensor格式
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)  # nested_tensor_from_tensor_list函数是给样本加mask操作
            # sample[mask(2,384,640),  (2,3,384,640)]
        features, pos = self.backbone(samples)  # backbone   position_encoding获取
        src, mask = features[-1].decompose()


        patches = patches.flatten(0, 1)
        patch_feature = self.backbone2(patches)  # [6,1024,8,8]# obtain feature maps for exemplar patches 提取样本特征
        patch_feature = self.EPF_extractor(patch_feature,
                                           scale_embedding)  # [3,N,256] # compress the feature maps into vectors and inject scale embeddings
        # 处理样本特征
        # Stage 2: enhance feature representation, e.g., the self similarity module.
        # refined_feature, patch_feature = self.refiner(features, patch_feature)

        # idx=66  num_object_queries
        bs = src.size(0)
        exe = patch_feature + self.query_embed.weight[:3, :].unsqueeze(1).repeat(1, bs, 1)
        hs = self.transformer(self.input_proj(src), mask, exe, pos[-1])[0]
        memory = self.transformer(self.input_proj(src), mask, exe, pos[-1])[1]#[N,256,h,w]

        hs = torch.mean(hs, dim=0)
        hs = hs.transpose(0, 1)# [N,3,256]
        # Stage 3: generate similarity map by densely measuring similarity.
        counting_feature, corr_map = self.matcher(memory, hs)
        # Stage 4: predicting density map
        density_map = self.counter(counting_feature)

        if not is_train:
            return density_map
        else:
            return {'corr_map': corr_map, 'density_map': density_map}

    # def _reset_parameters(self):
    #    for p in self.parameters():
    #        if p.dim() > 1:
    #            nn.init.xavier_uniform_(p)
