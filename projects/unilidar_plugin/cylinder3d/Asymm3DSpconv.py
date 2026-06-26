'''
Author: EASON XU
Date: 2025-05-29 02:22:42
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-06-03 02:04:10
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/cylinder3d/Asymm3DSpconv.py
'''
import torch
from torch import Tensor, nn
from spconv.pytorch.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
from typing import Optional, Union, List
from mmengine.config import ConfigDict
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
from mmcv.cnn import build_activation_layer, build_norm_layer
import numpy as np
from mmengine.model import BaseModule
from mmdet3d.models.builder import BACKBONES


class AsymmResBlock(nn.Module):
    """Asymmetrical Residual Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None):
        super().__init__()
        if indice_key is None:
            indice_key = 'res'

        self.conv0_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_0_0')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_0_1')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_1_0')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_1_1')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)
        shortcut = shortcut.replace_feature(self.act0_0(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_0(shortcut.features))

        shortcut = self.conv0_1(shortcut)
        shortcut = shortcut.replace_feature(self.act0_1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_1(shortcut.features))

        res = self.conv1_0(x)
        res = res.replace_feature(self.act1_0(res.features))
        res = res.replace_feature(self.bn1_0(res.features))

        res = self.conv1_1(res)
        res = res.replace_feature(self.act1_1(res.features))
        res = res.replace_feature(self.bn1_1(res.features))

        res = res.replace_feature(res.features + shortcut.features)

        return res


class AsymmeDownBlock(nn.Module):
    """Asymmetrical DownSample Block.

    Args:
       in_channels (int): Input channels of the block.
       out_channels (int): Output channels of the block.
       norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
       act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
       pooling (bool): Whether pooling features at the end of
           block. Defaults: True.
       height_pooling (bool): Whether pooling features at
           the height dimension. Defaults: False.
       indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 pooling: bool = True,
                 height_pooling: bool = False,
                 indice_key: Optional[str] = None):
        super().__init__()
        self.pooling = pooling
        if indice_key is None:
            indice_key = 'down'

        self.conv0_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_0_0')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_0_1')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_1_0')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_1_1')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

        if pooling:
            if height_pooling:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key=indice_key + '_pool',
                    bias=False)
            else:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(2, 2, 1),
                    padding=1,
                    indice_key=indice_key + '_pool',
                    bias=False)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)
        shortcut = shortcut.replace_feature(self.act0_0(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_0(shortcut.features))

        shortcut = self.conv0_1(shortcut)
        shortcut = shortcut.replace_feature(self.act0_1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_1(shortcut.features))

        res = self.conv1_0(x)
        res = res.replace_feature(self.act1_0(res.features))
        res = res.replace_feature(self.bn1_0(res.features))

        res = self.conv1_1(res)
        res = res.replace_feature(self.act1_1(res.features))
        res = res.replace_feature(self.bn1_1(res.features))

        res = res.replace_feature(res.features + shortcut.features)

        if self.pooling:
            pooled_res = self.pool(res)
            return pooled_res, res
        else:
            return res


class AsymmeUpBlock(nn.Module):
    """Asymmetrical UpSample Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
                normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
                Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
        up_key (str, optional): Name of indice tables used in
            SparseInverseConv3d. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None,
                 up_key: Optional[str] = None):
        super().__init__()
        if indice_key is None:
            indice_key = 'up'
        if up_key is None:
            up_key = indice_key + '_up'

        self.trans_conv = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_trans')
        self.trans_act = build_activation_layer(act_cfg)
        self.trans_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_conv1')
        self.act1 = build_activation_layer(act_cfg)
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv2 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_conv2')
        self.act2 = build_activation_layer(act_cfg)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv3 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_conv3')
        self.act3 = build_activation_layer(act_cfg)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]

        self.up_subm = SparseInverseConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            indice_key=up_key,
            bias=False)

    def forward(self, x: SparseConvTensor,
                skip: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        x_trans = self.trans_conv(x)
        x_trans = x_trans.replace_feature(self.trans_act(x_trans.features))
        x_trans = x_trans.replace_feature(self.trans_bn(x_trans.features))

        # upsample
        up = self.up_subm(x_trans)

        up = up.replace_feature(up.features + skip.features)

        up = self.conv1(up)
        up = up.replace_feature(self.act1(up.features))
        up = up.replace_feature(self.bn1(up.features))

        up = self.conv2(up)
        up = up.replace_feature(self.act2(up.features))
        up = up.replace_feature(self.bn2(up.features))

        up = self.conv3(up)
        up = up.replace_feature(self.act3(up.features))
        up = up.replace_feature(self.bn3(up.features))

        return up
    
class DDCMBlock(nn.Module):
    """Dimension-Decomposition based Context Modeling.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='Sigmoid').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='Sigmoid'),
                 indice_key: Optional[str] = None):
        super().__init__()
        if indice_key is None:
            indice_key = 'ddcm'

        self.conv1 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=1,
            bias=False,
            indice_key=indice_key + '_conv1')
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key + '_conv2')
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + '_conv3')
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act3 = build_activation_layer(act_cfg)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn1(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv2(x)
        shortcut2 = shortcut2.replace_feature(self.bn2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act2(shortcut2.features))

        shortcut3 = self.conv3(x)
        shortcut3 = shortcut3.replace_feature(self.bn3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act3(shortcut3.features))
        
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)
        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut

@BACKBONES.register_module()
class Asymm3DSpconv(BaseModule):
    """Asymmetrical 3D convolution networks.

    Args:
        grid_size (int): Size of voxel grids.
        input_channels (int): Input channels of the block.
        base_channels (int): Initial size of feature channels before
            feeding into Encoder-Decoder structure. Defaults to 16.
        backbone_depth (int): The depth of backbone. The backbone contains
            downblocks and upblocks with the number of backbone_depth.
        height_pooing (List[bool]): List indicating which downblocks perform
            height pooling.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01)).
        init_cfg (dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 grid_size: int,
                 input_channels: int,
                 base_channels: int = 16,
                 backbone_depth: int = 4,
                 height_pooing: List[bool] = [True, True, False, False],
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.grid_size = grid_size
        self.backbone_depth = backbone_depth
        self.down_context = AsymmResBlock(
            input_channels, base_channels, indice_key='pre', norm_cfg=norm_cfg)

        self.down_block_list = torch.nn.ModuleList()
        self.up_block_list = torch.nn.ModuleList()
        for i in range(self.backbone_depth):
            down_key = f'down_{i}'
            self.down_block_list.append(
                AsymmeDownBlock(
                    2**i * base_channels,
                    2**(i + 1) * base_channels,
                    height_pooling=height_pooing[i],
                    indice_key=down_key,
                    norm_cfg=norm_cfg))
            if i == self.backbone_depth - 1:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 1) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key=down_key + '_pool',
                        indice_key=f'up_{self.backbone_depth - 1 - i}',
                        norm_cfg=norm_cfg))
            else:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 2) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key=down_key + '_pool',
                        indice_key=f'up_{self.backbone_depth - 1 - i}',
                        norm_cfg=norm_cfg))

        self.ddcm = DDCMBlock(
            2 * base_channels,
            2 * base_channels,
            indice_key='ddcm',
            norm_cfg=norm_cfg)

    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> SparseConvTensor:
        """Forward pass."""
        coors = coors.int()
        ret = SparseConvTensor(voxel_features, coors, np.array(self.grid_size),
                               batch_size)
        ret = self.down_context(ret)

        down_skip_list = []
        down_pool = ret
        for i in range(self.backbone_depth):
            down_pool, down_skip = self.down_block_list[i](down_pool)
            down_skip_list.append(down_skip)

        up = down_pool
        for i in range(self.backbone_depth - 1, -1, -1):
            up = self.up_block_list[i](up, down_skip_list[i])

        ddcm = self.ddcm(up)
        ddcm = ddcm.replace_feature(torch.cat((ddcm.features, up.features), 1))

        return ddcm
