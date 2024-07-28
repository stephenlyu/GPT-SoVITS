# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
from typing import List
import torch
from tqdm import tqdm

from AR.modules.embedding_onnx import SinePositionalEmbedding
from AR.modules.embedding_onnx import TokenEmbedding
from AR.modules.transformer_onnx import LayerNorm
from AR.modules.transformer_onnx import TransformerEncoder
from AR.modules.transformer_onnx import TransformerEncoderLayer
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}
debug = False

inf_tensor_value = torch.FloatTensor([-float("Inf")]).float()

def logits_to_probs(
    logits,
    previous_tokens = None,
    temperature: float = 1.0,
    top_k = None,
    top_p = None,
    repetition_penalty: float = 1.0,
):
    global inf_tensor_value
    previous_tokens = previous_tokens.squeeze(0)
    # if previous_tokens is not None and repetition_penalty != 1.0:
    previous_tokens = previous_tokens.long()
    score = torch.gather(logits, dim=0, index=previous_tokens)
    score = torch.where(
        score < 0, score * repetition_penalty, score / repetition_penalty
    )
    logits.scatter_(dim=0, index=previous_tokens, src=score)

    # if top_p is not None and top_p < 1.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     print('sorted_logits:', sorted_logits.shape)
    #     cum_probs = torch.cumsum(
    #         torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
    #     )
    #     sorted_indices_to_remove = cum_probs > top_p
    #     sorted_indices_to_remove[0] = False  # keep at least one option
    #     indices_to_remove = sorted_indices_to_remove.scatter(
    #         dim=0, index=sorted_indices, src=sorted_indices_to_remove
    #     )
    #     logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    # if top_k is not None:
    v, _ = torch.topk(logits, top_k)
    pivot = v.select(-1, -1).unsqueeze(-1)
    if inf_tensor_value.device != logits.device:
        inf_tensor_value = inf_tensor_value.to(device=logits.device)
    logits = torch.where(logits < pivot, inf_tensor_value, logits)

    probs = torch.nn.functional.softmax(logits.unsqueeze(0), dim=1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort
):  # Does multinomial sampling without a cuda synchronization
    q = torch.randn_like(probs_sort)
    return torch.argmax(probs_sort / q, dim=1, keepdim=True).to(dtype=torch.int)[0]


def sample(
    logits,
    previous_tokens,
    **sampling_kwargs,
):
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


# @torch.jit.script
class T2SMLP:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

        w1.requires_grad = False
        b1.requires_grad = False
        w2.requires_grad = False
        b2.requires_grad = False

    def forward(self, x):
        x = F.relu(F.linear(x, self.w1, self.b1))
        x = F.linear(x, self.w2, self.b2)
        return x


# @torch.jit.script
class T2SBlock:
    def __init__(
        self,
        num_heads,
        hidden_dim: int,
        mlp: T2SMLP,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        norm_eps1,
        norm_w2,
        norm_b2,
        norm_eps2,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

        self.qkv_w.requires_grad = False
        self.qkv_b.requires_grad = False
        self.out_w.requires_grad = False
        self.out_b.requires_grad = False
        self.norm_w1.requires_grad = False
        self.norm_b1.requires_grad = False
        self.norm_w2.requires_grad = False
        self.norm_b2.requires_grad = False

    def process_prompt(self, x, attn_mask : torch.Tensor):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        k_cache = k
        v_cache = v

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size, -1, self.hidden_dim)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = F.layer_norm(
            x + attn, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = F.layer_norm(
            x + self.mlp.forward(x),
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache

    def decode_next_token(self, x, k_cache, v_cache):
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache, k], dim=1)
        v_cache = torch.cat([v_cache, v], dim=1)
        kv_len = k_cache.shape[1]

        batch_size = q.shape[0]
        q_len = q.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)


        attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size, -1, self.hidden_dim)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = F.layer_norm(
            x + attn, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = F.layer_norm(
            x + self.mlp.forward(x),
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


# @torch.jit.script
class T2STransformer:
    def __init__(self, num_blocks : int, blocks: List[T2SBlock]):
        self.num_blocks : int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self, x, attn_mask : torch.Tensor):
        k_cache : List[torch.Tensor] = []
        v_cache : List[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(x, attn_mask)
            k_cache.append(k_cache_.unsqueeze(0))
            v_cache.append(v_cache_.unsqueeze(0))
        k_cache = torch.concat(k_cache, dim=0)
        v_cache = torch.concat(v_cache, dim=0)
        return x, k_cache, v_cache

    def decode_next_token(
        self, x, k_cache: List[torch.Tensor], v_cache: List[torch.Tensor]
    ):
        k_cache_new : List[torch.Tensor] = []
        v_cache_new : List[torch.Tensor] = []

        for i in range(self.num_blocks):
            x, k, v = self.blocks[i].decode_next_token(x, k_cache[i], v_cache[i])
            k_cache_new.append(k.unsqueeze(0))
            v_cache_new.append(v.unsqueeze(0))
        k_cache = torch.concat(k_cache_new, dim=0)
        v_cache = torch.concat(v_cache_new, dim=0)
        return x, k_cache, v_cache

class OnnxEncoder(nn.Module):
    def __init__(self, ar_text_embedding, bert_proj, ar_text_position):
        super().__init__()
        self.ar_text_embedding = ar_text_embedding
        self.bert_proj = bert_proj
        self.ar_text_position = ar_text_position
    
    def forward(self, x, bert_feature):
        if debug:
            print('all_phoneme_ids:', x.shape)
            print('bert:', bert_feature.shape)
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        return self.ar_text_position(x)


class T2SFirstStageDecoder(nn.Module):
    def __init__(self, ar_audio_embedding, ar_audio_position, t2s_transformer:T2STransformer, ar_predict_layer, loss_fct, ar_accuracy_metric,
    top_k, num_layers):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
        self.t2s_transformer = t2s_transformer
        self.ar_predict_layer = ar_predict_layer
        self.loss_fct = loss_fct
        self.ar_accuracy_metric = ar_accuracy_metric
        self.top_k = top_k
        self.num_layers = num_layers
    
    def forward(self, x, prompt):
        y = prompt
        x_example = x[:,:,0] * 0.0

        y_emb = self.ar_audio_embedding(y)        

        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        y_example = y_pos[:,:,0] * 0.0
        x_attn_mask = torch.matmul(x_example.transpose(0, 1) , x_example).bool()
        y_attn_mask = torch.ones_like(torch.matmul(y_example.transpose(0, 1), y_example), dtype=torch.int64)
        y_attn_mask = torch.cumsum(y_attn_mask, dim=1) - torch.cumsum(
            torch.ones_like(y_example.transpose(0, 1), dtype=torch.int64), dim=0
        )
        y_attn_mask = y_attn_mask > 0

        x_y_pad = torch.matmul(x_example.transpose(0, 1), y_example).bool()
        y_x_pad = torch.matmul(y_example.transpose(0, 1), x_example).bool()
        x_attn_mask_pad = torch.cat([x_attn_mask, torch.ones_like(x_y_pad)], dim=1)
        y_attn_mask = torch.cat([y_x_pad, y_attn_mask], dim=1)
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)

        xy_dec, k_cache, v_cache = self.t2s_transformer.process_prompt(xy_pos, xy_attn_mask)

        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

        y = torch.concat([y, samples], dim=1)

        if debug:
            print('T2SFirstStageDecoder.forward')
            print('x:', x.shape)
            print('x_example:', x_example.shape)
            print('y:', y.shape)
            print('y_emb:', y_emb.shape)
            print('xy_pos:', xy_pos.shape)
            print('y_attn_mask:', y_attn_mask.shape)
            print('xy_attn_mask:', xy_attn_mask.shape)
            print('xy_dec:', xy_dec.shape)
            print('logits:', logits.shape)
            print('samples:', samples.shape)

        return y, k_cache, v_cache, y_emb


class T2SStageDecoder(nn.Module):
    def __init__(self, ar_audio_embedding, ar_audio_position, t2s_transformer:T2STransformer, ar_predict_layer, loss_fct, ar_accuracy_metric,
    top_k, num_layers):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
        self.t2s_transformer = t2s_transformer
        self.ar_predict_layer = ar_predict_layer
        self.loss_fct = loss_fct
        self.ar_accuracy_metric = ar_accuracy_metric
        self.top_k = top_k
        self.num_layers = num_layers

    def forward(self, y, k, v, y_emb):
        y_emb = torch.cat(
            [y_emb, self.ar_audio_embedding(y[:, -1:])], 1
        )
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = y_pos[:, -1:]
        
        xy_dec, k_cache, v_cache = self.t2s_transformer.decode_next_token(xy_pos, k, v)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = sample(logits[0], y, top_k=self.top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)

        y = torch.concat([y, samples], dim=1)

        if debug:
            print('T2SStageDecoder.forward')
            print('y:', y.shape)
            print('k:', k.shape)
            print('v:', v.shape)
            print('y_emb:', y_emb.shape)
            print('xy_pos:', xy_pos.shape)
            print('xy_dec:', xy_dec.shape)
            print('logits:', logits.shape)
            print('samples:', samples.shape)

        return y, k_cache, v_cache, y_emb, logits, samples


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = float(config["model"]["dropout"])
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )
        self.top_k = torch.LongTensor([5])
        self.early_stop_num = torch.LongTensor([-1])

        blocks = []

        for i in range(self.num_layers):
            layer = self.h.layers[i]
            t2smlp = T2SMLP(
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias
            )
            # (layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps
            )

            blocks.append(block)
        
        self.t2s_transformer = T2STransformer(self.num_layers, blocks)

    def init_onnx(self):
        self.onnx_encoder = OnnxEncoder(self.ar_text_embedding, self.bert_proj, self.ar_text_position)
        self.first_stage_decoder = T2SFirstStageDecoder(self.ar_audio_embedding, self.ar_audio_position, self.t2s_transformer, 
            self.ar_predict_layer, self.loss_fct, self.ar_accuracy_metric, self.top_k, 
            self.num_layers)
        self.stage_decoder = T2SStageDecoder(self.ar_audio_embedding, self.ar_audio_position, self.t2s_transformer, 
            self.ar_predict_layer, self.loss_fct, self.ar_accuracy_metric, self.top_k, 
            self.num_layers)

    def forward(self, x, prompts, bert_feature):
        early_stop_num = self.early_stop_num
        prefix_len = prompts.shape[1]
        if debug:
            print('early_stop_num:', self.early_stop_num)
            print('top_k:', self.top_k)

            import numpy as np
            np.save('TEMP/onnx-x.npy', x.cpu().numpy())
            np.save('TEMP/onnx-prompts.npy', prompts.cpu().numpy())
            np.save('TEMP/onnx-bert.npy', bert_feature.cpu().numpy())
        x = self.onnx_encoder(x, bert_feature)

        y, k, v, y_emb = self.first_stage_decoder(x, prompts)

        stop = False
        times = []
        import time
        for idx in range(1500):
            start = time.time()
            enco = self.stage_decoder(y, k, v, y_emb)
            times.append(time.time() - start)
            y, k, v, y_emb, logits, samples = enco
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                break
        print(times)
        y[0, -1] = 0
        return y, idx

    def infer_panel(
        self,
        x,  #####全部文本token
        x_lens,
        prompts,  ####参考音频token
        bert_feature,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        self.top_k[0] = top_k
        self.early_stop_num[0] = early_stop_num
        return self.forward(x, prompts, bert_feature)

    def infer(self, x, prompts, bert_feature):
        top_k = self.top_k
        early_stop_num = self.early_stop_num

        x = self.onnx_encoder(x, bert_feature)

        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]
        x_example = x[:,:,0] * 0.0
        x_attn_mask = torch.matmul(x_example.transpose(0, 1), x_example)
        x_attn_mask = torch.zeros_like(x_attn_mask, dtype=torch.bool)

        stop = False
        cache = {
            "all_stage": self.num_layers,
            "k": [None] * self.num_layers,
            "v": [None] * self.num_layers,
            "y_emb": None,
            "first_infer": 1,
            "stage": 0,
        }
        for idx in range(1500):
            if cache["first_infer"] == 1:
                y_emb = self.ar_audio_embedding(y)
            else:
                y_emb = torch.cat(
                    [cache["y_emb"], self.ar_audio_embedding(y[:, -1:])], 1
                )
            cache["y_emb"] = y_emb
            y_pos = self.ar_audio_position(y_emb)
            if cache["first_infer"] == 1:
                xy_pos = torch.concat([x, y_pos], dim=1)
            else:
                xy_pos = y_pos[:, -1:]
            y_len = y_pos.shape[1]
            if cache["first_infer"] == 1:
                x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
                y_attn_mask = F.pad(
                    torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
                    (x_len, 0), value=False
                )
                xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            else:
                xy_attn_mask = torch.zeros((1, x_len + y_len), dtype=torch.bool)
            xy_dec = self.h(xy_pos, mask=xy_attn_mask, cache=cache)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = sample(logits[0], y, top_k=top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                stop = True
            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                break
            y = torch.concat([y, samples], dim=1)
            cache["first_infer"] = 0
        return y, idx