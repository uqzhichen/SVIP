"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import pickle


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # vit输入图片的大小必须要固定的
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim768]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]   为了方便计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features = in_features*4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, attn_map = self.attn(norm_x)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_map


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, keep_token=15, pool='max', sim_score='cos', replace_n=1, dataset='CUB'):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.dataset =dataset
        self.keep_token = keep_token
        if pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
        self.replace_n = replace_n
        self.sim_score = sim_score

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        self.head_1 = nn.Linear(embed_dim*4, self.num_classes)
        # self.head_2 = nn.Linear(embed_dim, 312)
        # self.fill_token = nn.Parameter(torch.randn(1, 3072), requires_grad=True)

        self.token_cls = nn.Sequential(
            nn.Linear(768*4, 512), #512
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        w2v_file =  "/home/zhi//Projects/SCViP_ZSL-main/attribute/w2v/"+self.dataset+"_attribute.pkl"
        with open(w2v_file, 'rb') as f:
            w2v = pickle.load(f)
        self.w2v_att = torch.from_numpy(w2v).float().cuda()  # 312 * 300
        self.v2p = nn.Linear(self.w2v_att.shape[1], 768)
        self.p2t = nn.Linear(self.w2v_att.shape[0], 4)
        # a

    def not_forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        '''
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

        '''
        return x


    def compute_attention_rollout(self, attn_maps):
        rollout = None
        for attn in attn_maps:
            attn_avg = attn.mean(dim=1)
            B, N, _ = attn_avg.shape
            I = torch.eye(N, device=attn_avg.device).unsqueeze(0).expand(B, -1, -1)
            attn_aug = attn_avg + I 
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True) # normalize
            if rollout is None:
                rollout = attn_aug
            else:
                rollout = torch.bmm(attn_aug, rollout)
        return rollout

    def forward(self, images, labels=None, semantics=None, retrieved_samples=None, inference=False, epoch=0):
        if self.training:
            B = images.shape[0]
            # x_query = images[:, 0]
            x_query = images

            x_query, patch_pred, attn_maps = self.forward_source_branch(x_query) # attn_maps: 12*[64, 12, 197, 197]
            att_query = x_query[:, 0]
            att_fm = x_query[:, 1:]

            att_fm = att_fm.reshape(B, 7, 2, 7, 2, 768).transpose(2, 3).reshape(B, 7, 7, -1).view(B, 49, 4*768)

            if self.sim_score == 'cos':
                prototype_query = self.pool(att_fm.transpose(2, 1)).transpose(2, 1)
                sim_query = prototype_query @ att_fm.transpose(-1, -2) # [64, 1, 49]
            elif self.sim_score == 'attn':
                sim_query = attn_maps[-1].mean(1)[:, 0, 1:].unsqueeze(1) # [64, 1, 196]
                sim_query = sim_query.view(B, 1, 7, 7, 4).mean(-1).flatten(2) # [64, 1, 49]
            elif self.sim_score == 'attn_rollout':
                rollout = self.compute_attention_rollout(attn_maps)  # rollout shape: [64, 197, 197]
                sim_query = rollout[:, 0, 1:].unsqueeze(1)  # [64, 1, 196]
                sim_query = sim_query.view(B, 1, 7, 7, 4).mean(-1).flatten(2)  # [64, 1, 49]

            # sim_query = att_query.unsqueeze(1) @ att_fm.transpose(-1, -2)
            pruning_idx_query = torch.sort(sim_query.topk(self.keep_token)[1])[0].transpose(-1, -2)
            index_expanded = pruning_idx_query.expand(-1, -1, att_fm.size(2))
            result_query = torch.gather(att_fm, 1, index_expanded)
            source_att_fm = self.pool(self.head_1(result_query).transpose(2, 1)).squeeze()
            # source_att_head = self.head_1(att_query)

            patch_labels = torch.ones(len(pruning_idx_query), 49).cuda()
            patch_labels.scatter_(1, pruning_idx_query.squeeze(), 0)

            # x_support = images[:, 1]
            # pruning_idx_support, mutual_matches = self.forward_target_branch(x_support, result_query)
            # pruned_output = self.forward_features(images, pruning_idx=[pruning_idx_query, pruning_idx_support],
            #                                       mutual_matches=mutual_matches)
            pruned_output = self.forward_features(images, pruning_idx=pruning_idx_query)

            # att_query = pruned_output[:, 0]
            pruned_fm = pruned_output[:, 1:]
            pruned_fm = pruned_fm.reshape(B, 7, 2, 7, 2, 768).transpose(2, 3).reshape(B, 7, 7, -1).view(B, 49, 4*768)
            # pruned_att = pruned_fm.mean(1)
            pruned_att = self.pool(pruned_fm.transpose(2, 1)).transpose(2, 1)

            # pruned_att = pruned_output[:, 0]
            # pruned_fm = pruned_output[:, 1:]

            sim_support = pruned_att @ pruned_fm.transpose(-1, -2)
            pruning_idx_support = torch.sort(sim_support.topk(self.keep_token)[1])[0].transpose(-1, -2)
            index_expanded = pruning_idx_support.expand(-1, -1, pruned_fm.size(2))
            result_support = torch.gather(pruned_fm, 1, index_expanded)

            pruned_att_fm = self.pool(self.head_1(result_support).transpose(2, 1)).squeeze()
            # pruned_att_head = self.head_1(pruned_att)
            return source_att_fm, pruned_att_fm, patch_labels, patch_pred
        else:
            pruned_output = self.forward_inference(images)

            return pruned_output


    def forward_source_branch(self, x):
        x_patches = self.patch_embed(x)
        B = x_patches.shape[0] # 49*49
        x_patches = x_patches.reshape(B, 7, 2, 7, 2, 768).transpose(2, 3).reshape(B, 7, 7, -1).view(B, 49, 4 * 768)
        patch_pred = torch.sigmoid(self.token_cls(x_patches).squeeze())
        x_patches = x_patches.view(B, 7, 7, 2, 2, 768).transpose(2, 3).reshape(B, 196, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x_patches), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_maps = []  # Store attention maps if needed
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map)

        x = self.norm(x)
        return x, patch_pred, attn_maps

    @torch.no_grad()
    def forward_target_branch(self, x, result_query):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)

        # prototype_support = self.pool(x.transpose(2, 1)).transpose(2, 1)
        att_support = x[:, 0].unsqueeze(1)
        att_fm = x[:, 1:]

        sim_support = att_support @ att_fm.transpose(-1, -2)
        pruning_idx_support = torch.sort(sim_support.topk(self.keep_token)[1])[0].transpose(-1, -2)
        index_expanded = pruning_idx_support.expand(-1, -1, x.size(2))
        result_support = torch.gather(att_fm, 1, index_expanded)

        sim1 = result_query @ result_support.transpose(-1, -2)
        sim2 = result_support @ result_query.transpose(-1, -2)
        max_indices_sim1 = torch.argmax(sim1, dim=-1)  # Shape: (B, 15)
        max_indices_sim2 = torch.argmax(sim2, dim=-1)  # Shape: (B, 15)
        # Create boolean tensors indicating where each index points
        # Create a one-hot encoded tensor to mark the positions of max indices in `sim1`
        one_hot_sim1 = torch.nn.functional.one_hot(max_indices_sim1, num_classes=self.keep_token)  # Shape: (B, 15, 15)
        # Create a one-hot encoded tensor to mark the positions of max indices in `sim2`
        one_hot_sim2 = torch.nn.functional.one_hot(max_indices_sim2, num_classes=self.keep_token)  # Shape: (B, 15, 15)
        # Transpose one_hot_sim2 to align the dimensions for comparison
        one_hot_sim2_transposed = one_hot_sim2.transpose(1, 2)  # Shape: (B, 15, 15)
        # Create a mutual match mask:
        # For each pair (i, j), check if both (i -> j) and (j -> i) hold true
        mutual_matches = one_hot_sim1 & one_hot_sim2_transposed  # Shape: (B, 15, 15)
        return pruning_idx_support, mutual_matches

    def forward_features(self, images, pruning_idx=None, mutual_matches=None):

        x_patches = self.patch_embed(images)

        B = x_patches.shape[0]
        query_idx = pruning_idx
        x_patches = x_patches.reshape(B, 7, 2, 7, 2, 768).transpose(2, 3).reshape(B, 7, 7, -1).view(B, 49, 4 * 768)

        ############
        self.fill_token = self.p2t(self.v2p(self.w2v_att).transpose(-2,-1)).reshape(1, 3072)
        ##############


        fill_tokens = self.fill_token.expand(B, 49 - self.keep_token, -1)
        tensor = query_idx.squeeze(-1)
        all_indices = torch.arange(0, 49).unsqueeze(0).repeat(tensor.size(0), 1).cuda() # Shape: (64, 196)
        mask = torch.ones_like(all_indices, dtype=torch.bool).cuda()
        mask.scatter_(1, tensor, False)  # Mark the indices that are present in the original tensor as False
        inverted_indices = all_indices[mask].reshape(tensor.size(0), -1, 1)  # Shape: (64, 49-15, 1)
        x_patches = x_patches.scatter_add_(1, inverted_indices.expand(-1, -1, x_patches.size(2)), fill_tokens) #scatter_add_
        x = x_patches.view(B, 7, 7, 2, 2, 768).transpose(2, 3).reshape(B, 196, 768)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)

        return x

    def forward_features1(self, images, pruning_idx=None, mutual_matches=None):

        x_patches = self.patch_embed(images[:, 0])
        B = x_patches.shape[0]

        with torch.no_grad():
            x_support = self.patch_embed(images[:, 1])

        query_idx = pruning_idx[0]
        support_idx = pruning_idx[1]

        fill_tokens = self.fill_token.expand(B, 196 - self.keep_token, -1)

        tensor = query_idx.squeeze(-1)
        all_indices = torch.arange(0, 196).unsqueeze(0).repeat(tensor.size(0), 1).cuda() # Shape: (64, 196)
        mask = torch.ones_like(all_indices, dtype=torch.bool).cuda()
        mask.scatter_(1, tensor, False)  # Mark the indices that are present in the original tensor as False
        inverted_indices = all_indices[mask].reshape(tensor.size(0), -1, 1)  # Shape: (64, 49-15, 1)
        query = x_patches
        query = query.scatter_add_(1, inverted_indices.expand(-1, -1, x_patches.size(2)), fill_tokens) #scatter_add_
        support = x_support
        # query = x_query
        for i, sample in enumerate(mutual_matches):
            num = 0
            for n, row in enumerate(sample):
                for m, column in enumerate(row):
                    if column == 1:
                        num += 1
                        query[i][query_idx[i][n]] = support[i][support_idx[i][m]]
                        if num == self.replace_n:
                            break
        x_patches = query

        x = x_patches
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)

        return x

    def forward_inference(self, x):
        x_patches = self.patch_embed(x)

        B = x_patches.shape[0]
        x_patches = x_patches.reshape(B, 7, 2, 7, 2, 768).transpose(2, 3).reshape(B, 7, 7, -1).view(B, 49, 4 * 768)
        patch_pred = torch.sigmoid(self.token_cls(x_patches)).squeeze().topk(self.keep_token, largest=False)[1]
        patch_pred = patch_pred.sort()[0]

        ############
        fill_token = self.p2t(self.v2p(self.w2v_att).transpose(-2,-1)).reshape(1, 3072)
        ##############

        fill_tokens = fill_token.expand(B, 49 - self.keep_token, -1)
        tensor = patch_pred.squeeze(-1)
        all_indices = torch.arange(0, 49).unsqueeze(0).repeat(tensor.size(0), 1).cuda() # Shape: (64, 196)
        mask = torch.ones_like(all_indices, dtype=torch.bool).cuda()
        mask.scatter_(1, tensor, False)  # Mark the indices that are present in the original tensor as False
        inverted_indices = all_indices[mask].reshape(tensor.size(0), -1, 1)  # Shape: (64, 49-15, 1)
        x_patches = x_patches.scatter_add_(1, inverted_indices.expand(-1, -1, x_patches.size(2)), fill_tokens) #scatter_add_
        x = x_patches.view(B, 7, 7, 2, 2, 768).transpose(2, 3).reshape(B, 196, 768)



        # B = x_patches.shape[0]
        # patch_pred = torch.sigmoid(self.token_cls(x_patches)).squeeze().topk(self.keep_token, largest=False)[1]
        # patch_pred = patch_pred.sort()[0]
        # fill_tokens = self.fill_token.expand(B, self.keep_token, -1)
        # tensor = patch_pred.squeeze(-1)
        # all_indices = torch.arange(0, 196).unsqueeze(0).repeat(tensor.size(0), 1).cuda() # Shape: (64, 196)
        # mask = torch.ones_like(all_indices, dtype=torch.bool).cuda()
        # mask.scatter_(1, tensor, False)  # Mark the indices that are present in the original tensor as False
        # inverted_indices = all_indices[mask].reshape(tensor.size(0), -1, 1)  # Shape: (64, 49-15, 1)
        # query = x_patches.scatter_add_(1, inverted_indices.expand(-1, -1, x_patches.size(2)), fill_tokens) #scatter_add_
        # x = query

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)

        pruned_fm = x[:, 1:]
        pruned_fm = pruned_fm.reshape(B, 7, 2, 7, 2, 768).transpose(2, 3).reshape(B, 7, 7, -1).view(B, 49, 4 * 768)
        # pruned_att = pruned_fm.mean(1)
        #
        # sim_support = pruned_att.unsqueeze(1) @ pruned_fm.transpose(-1, -2)
        # pruning_idx_support = torch.sort(sim_support.topk(self.keep_token)[1])[0].transpose(-1, -2)
        # index_expanded = pruning_idx_support.expand(-1, -1, pruned_fm.size(2))
        # x = torch.gather(pruned_fm, 1, index_expanded)

        # sim_support = x[:, 0].unsqueeze(1) @ x[:,1:].transpose(-1, -2)
        # pruning_idx_support = torch.sort(sim_support.topk(self.keep_token)[1])[0].transpose(-1, -2)
        # index_expanded = pruning_idx_support.expand(-1, -1, x.size(2))
        # result_query = torch.gather(x[:,1:], 1, index_expanded)
        #
        # index_expanded = patch_pred.unsqueeze(-1).expand(-1, -1, x.size(2))
        # result_query = torch.gather(x[:,1:], 1, index_expanded)

        # prototype_re = self.pool(x.transpose(2, 1)).transpose(2, 1)
        #
        # sim_re = prototype_re @ x.transpose(-1, -2)
        # pruning_idx_query = torch.sort(sim_re.topk(self.keep_token)[1])[0].transpose(-1, -2)
        # index_expanded = pruning_idx_query.expand(-1, -1, x.size(2))
        # result_re = torch.gather(x, 1, index_expanded)
        # pruned_att = self.pool(self.head_1(x).transpose(2, 1)).squeeze()
        return self.pool(self.head_1(pruned_fm).transpose(2, 1)).squeeze()

        # return self.head_1(x[:, 0])


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

