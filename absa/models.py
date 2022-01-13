from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch
import math
from collections import Iterable
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig, RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaForSequenceClassification, RobertaTokenizer

class BertMultiAspectsCLS(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiAspectsCLS, self).__init__(config)
        self.num_labels = self.config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size, self.num_labels)

        self.attn_mlp = nn.Linear(config.hidden_size, 1)
        self.linear_map = nn.Linear(config.hidden_size * 3, config.hidden_size)

    def classifier(self, emb):
        cls_feature = self.dropout(emb)
        logits = self.proj(cls_feature)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                multi_labels=None, multi_aspects_pos=None, *args, **kwargs):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)
        # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        bs = len(input_ids)
        aspects_emb = []
        labels = []
        aspects_nums = []  # the number of aspects in each sentence
        for _i in range(bs):
            _j = 0
            while _j < len(multi_labels[_i]) and multi_labels[_i][_j] != -1:
                labels.append(multi_labels[_i][_j])
                asp_from = multi_aspects_pos[_i][_j * 2] + 1   # the first token is [CLS]
                asp_to = multi_aspects_pos[_i][_j*2 + 1] + 1
                # aspects_emb.append(torch.mean(sequence_output[_i][asp_from:asp_to],dim=0))
                # assert list(token_type_ids[_i][asp_from:asp_to]) == [1] * (int(asp_to)-int(asp_from))
                # ------ get enhanced span embedding -----
                tmp_emb = torch.cat((sequence_output[_i][asp_from], sequence_output[_i][asp_to - 1]), dim=0)
                attn_weight = self.attn_mlp(sequence_output[_i][asp_from:asp_to])
                attn_weight = torch.softmax(attn_weight, dim=0).expand(-1, len(sequence_output[_i][asp_from]))
                attn_emb = torch.sum(attn_weight * sequence_output[_i][asp_from:asp_to], dim=0)
                aspects_emb.append(self.linear_map(torch.cat((tmp_emb, attn_emb), dim=0)))
                # -----------------------------------------

                _j += 1
            aspects_nums.append(_j)

        labels = torch.tensor(labels, dtype=torch.long, device=input_ids.device)
        aspects_emb = torch.stack(aspects_emb, dim=0)
        assert len(labels) == len(aspects_emb) == sum(aspects_nums)

        logits = self.classifier(aspects_emb)
        cls_loss_fn = nn.CrossEntropyLoss()
        cls_loss = cls_loss_fn(logits,labels)

        # regular
        regular_term = 0 #
        total_times = 1
        # ---------  just for different aspects ------------
        for _idx, _num in enumerate(aspects_nums):
            if _num > 1:
                _tmp = sum(aspects_nums[:_idx])
                for _i in range(_tmp, _tmp + _num - 1):
                    for _j in range(_i + 1, _tmp + _num):
                        if False or labels[_i] != labels[_j]: # do regular when they have different polarity
                            # regular_term += torch.abs(torch.mean(aspects_emb[_i] * aspects_emb[_j]))
                            regular_term += torch.abs(torch.cosine_similarity(aspects_emb[_i], aspects_emb[_j], dim=-1))
                            total_times += 1
                        else:
                            regular_term += 1 - torch.cosine_similarity(aspects_emb[_i], aspects_emb[_j], dim=-1)
                            total_times += 1

        outputs = [cls_loss, logits, labels]
        return tuple(outputs)

class RoBERTaMultiAspectsCLS(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(RoBERTaMultiAspectsCLS, self).__init__(config)
        self.num_labels = self.config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size, self.num_labels)

        self.attn_mlp = nn.Linear(config.hidden_size, 1)
        self.linear_map = nn.Linear(config.hidden_size * 3, config.hidden_size)

    def classifier(self, emb):
        cls_feature = self.dropout(emb)
        logits = self.proj(cls_feature)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                multi_labels=None, multi_aspects_pos=None, *args, **kwargs):
        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=None,
                            position_ids=position_ids)
        # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        bs = len(input_ids)
        aspects_emb = []
        labels = []
        aspects_nums = []  # the number of aspects in each sentence
        for _i in range(bs):
            _j = 0
            while _j < len(multi_labels[_i]) and multi_labels[_i][_j] != -1:
                labels.append(multi_labels[_i][_j])
                asp_from = multi_aspects_pos[_i][_j * 2] + 1   # the first token is [CLS]
                asp_to = multi_aspects_pos[_i][_j*2 + 1] + 1
                # aspects_emb.append(torch.mean(sequence_output[_i][asp_from:asp_to],dim=0))
                # assert list(token_type_ids[_i][asp_from:asp_to]) == [1] * (int(asp_to)-int(asp_from))
                # ------ get enhanced span embedding -----
                tmp_emb = torch.cat((sequence_output[_i][asp_from], sequence_output[_i][asp_to - 1]), dim=0)
                attn_weight = self.attn_mlp(sequence_output[_i][asp_from:asp_to])
                attn_weight = torch.softmax(attn_weight, dim=0).expand(-1, len(sequence_output[_i][asp_from]))
                attn_emb = torch.sum(attn_weight * sequence_output[_i][asp_from:asp_to], dim=0)
                aspects_emb.append(self.linear_map(torch.cat((tmp_emb, attn_emb), dim=0)))
                # -----------------------------------------

                _j += 1
            aspects_nums.append(_j)

        labels = torch.tensor(labels, dtype=torch.long, device=input_ids.device)
        aspects_emb = torch.stack(aspects_emb, dim=0)
        assert len(labels) == len(aspects_emb) == sum(aspects_nums)

        logits = self.classifier(aspects_emb)
        cls_loss_fn = nn.CrossEntropyLoss()
        cls_loss = cls_loss_fn(logits,labels)

        # regular
        regular_term = 0 #
        total_times = 1
        # ---------  just for different aspects ------------
        for _idx, _num in enumerate(aspects_nums):
            if _num > 1:
                _tmp = sum(aspects_nums[:_idx])
                for _i in range(_tmp, _tmp + _num - 1):
                    for _j in range(_i + 1, _tmp + _num):
                        if False or labels[_i] != labels[_j]: # do regular when they have different polarity
                            # regular_term += torch.abs(torch.mean(aspects_emb[_i] * aspects_emb[_j]))
                            regular_term += torch.abs(torch.cosine_similarity(aspects_emb[_i], aspects_emb[_j], dim=-1))
                            total_times += 1
                        else:
                            regular_term += 1 - torch.cosine_similarity(aspects_emb[_i], aspects_emb[_j], dim=-1)
                            total_times += 1

        outputs = [cls_loss, logits, labels]
        return tuple(outputs)

class RoBERTaSpanCLS(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(RoBERTaSpanCLS, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = self.make_ffnn(3 * config.hidden_size, 3 * config.hidden_size, self.num_labels)
        self.attn_mlp = nn.Linear(config.hidden_size, 1)

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        return nn.Linear(in_features, out_features, bias)

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def classifier(self, emb):
        cls_feature = self.dropout(emb)
        logits = self.cls(cls_feature)
        return logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        input_masks = attention_mask.to(torch.bool)
        bs = input_ids.shape[0]

        final_embs_list = []
        for bs_idx in range(bs):
            valid_embs = sequence_output[bs_idx][input_masks[bs_idx]][1:-1] # the first token is [CLS] and the last token is [SEP]

            head_emb = valid_embs[0]
            if valid_embs.shape[0] == 1:
                final_emb = torch.cat([head_emb,head_emb,head_emb], dim=0)
            else:
                tail_emb = valid_embs[-1]
                tokens_attn = torch.squeeze(self.attn_mlp(valid_embs))  # [num_wps]
                tokens_attn = nn.functional.softmax(tokens_attn)  # [num_asp+num_spans, num_wps]

                attn_emb = torch.matmul(tokens_attn, valid_embs)

                final_emb = torch.cat([head_emb, tail_emb, attn_emb], dim=0)
            final_embs_list.append(final_emb)


        final_embs = torch.stack(final_embs_list, dim=0) # [bs, hidden_size * 3]

        logits = self.classifier(final_embs)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class RoBERTaSpanCLSAdvDistillation(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(RoBERTaSpanCLSAdvDistillation, self).__init__(config)
        self.num_labels = self.config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.attn_mlp = nn.Linear(config.hidden_size, 1)

        self.save_spans_info = config.save_spans_info
        self.use_gate = config.use_gate
        self.no_distillation = config.no_distillation
        self.no_sentiment_score = config.no_sentiment_score
        self.no_adversarial = config.no_adversarial
        self.no_adv_loss = config.no_adv_loss
        self.adversarial_on_polarity = config.adversarial_on_polarity


        self.top_n = config.top_n  # top_n spans according to the total score
        self.max_span_width = config.max_span_width
        self.use_width_features = config.use_width_features
        self.use_rela_dis_features = config.use_rela_dis_features
        self.use_dep_dis_features = config.use_dep_dis_features
        self.feature_emb_size = config.feature_emb_size
        self.span_emb_size = config.hidden_size * 3 + self.feature_emb_size if self.use_width_features else config.hidden_size * 3
        self.max_rela_distance = config.max_rela_distance
        self.adv_loss_weight = config.adv_loss_weight
        self.max_dep_dis_distance = 20
        self.pair_emb_size = self.span_emb_size * 3  + self.feature_emb_size if (self.use_rela_dis_features or self.use_dep_dis_features) else self.span_emb_size * 3

        self.absa_cls = self.make_ffnn(self.span_emb_size, self.span_emb_size, self.num_labels)

        self.emb_span_width = nn.Embedding(self.max_span_width, self.feature_emb_size) if self.use_width_features else None
        self.emb_rela_distance = nn.Embedding(self.max_rela_distance, self.feature_emb_size) if self.use_rela_dis_features else None
        self.emb_dep_dis_distance = nn.Embedding(self.max_dep_dis_distance, self.feature_emb_size) if self.use_dep_dis_features else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, 3000, output_size=3)
        self.dependency_score_ffnn = self.make_ffnn(self.pair_emb_size, 3000, output_size=1) # TODO: dot product/ change mlp into bert layer/ multi-head
        self.gate_ffnn = self.make_ffnn(self.span_emb_size * 2, 3000, output_size=self.span_emb_size)


        config.span_emb_size = self.span_emb_size

        # discriminator: to classify the polarity of aspect
        # single dense layer
        self.nums_bias_label = config.nums_bias_label

        # two-layer MLP

        self.disc = self.make_ffnn(self.span_emb_size, self.span_emb_size, self.nums_bias_label)

    def get_params(self, named=True):
        # encoder parameters
        encoder_params = []
        for name, param in self.named_parameters():
            if name.startswith('bert') or name.startswith('roberta'):
                to_add = (name, param) if named else param
                encoder_params.append(to_add)

        # task-based parameters
        task_params = list(list(self.absa_cls.parameters()) + \
            list(self.attn_mlp.parameters()) + list(self.span_emb_score_ffnn.parameters()) + \
            list(self.dependency_score_ffnn.parameters()) + list(self.gate_ffnn.parameters()))
        if self.use_width_features:
            task_params += list(self.emb_span_width.parameters())
        if self.use_rela_dis_features:
            task_params += list(self.emb_rela_distance.parameters())
        if self.use_dep_dis_features:
            task_params += list(self.emb_dep_dis_distance.parameters())

        disc_params = self.disc.parameters()

        return encoder_params, task_params, disc_params

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        return nn.Linear(in_features, out_features, bias)

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def classifier(self, emb):
        cls_feature = self.dropout(emb)
        logits = self.absa_cls(cls_feature)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def discriminator(self, emb):
        # Note: detach the embedding since when don't want the gradient to flow
        #       all the way to the encoder. disc_loss is used only to change the
        #       parameters of the discriminator network
        # if self.sharing_all:
        #     pure_logits = self.absa_cls(self.dropout(emb.detach()))
        # else:
        pure_logits = self.disc(self.dropout(emb.detach()))
        return pure_logits

    def adv_classifier(self, emb):
        # Note: detach the embedding since when don't want the gradient to flow
        #       all the way to the encoder. disc_loss is used only to change the
        #       parameters of the discriminator network
        # if self.sharing_all:
        #     pure_logits = self.absa_cls(self.dropout(emb))
        # else:
        pure_logits = self.disc(self.dropout(emb))
        return pure_logits

    def refine_aspect(self, asp_embs, span_embs, scores, max_num=None, dummy_scores=None):
        # asp_embs: [num_asp, emb_size]
        # span_embs: [num_spans, emb_size]
        # scores: [num_asp, num_spans]

        # add dummy scores and get attention weights
        num_asp = asp_embs.shape[0]
        # dummy_scores = torch.full((num_asp, 1), fill_value=9,device=scores.device) # the dummy span's score
        # scores = torch.cat([dummy_scores, scores], dim=1)
        if dummy_scores is not None:
            scores = torch.cat([dummy_scores, scores], dim=1)
        else:
            scores = torch.cat([torch.zeros(num_asp, 1, device=scores.device), scores], dim=1) # [num_asp, 1 + num_spans]
        attn_weights = nn.functional.softmax(scores, dim=-1) # [num_asp, 1 + num_spans]

        # get the new asp_embs with attention weight
        span_embs = torch.unsqueeze(span_embs, 0).repeat(num_asp, 1, 1)  # [num_asp, num_spans, emb_size]
        # if self.use_explict_dummy_span:
        #     dummy_spans = self.dummy_span_emb(torch.tensor([0], dtype=torch.long, device=asp_embs.device)) # [1, emb_size]
        #     dummy_spans = dummy_spans.repeat(num_asp, 1) # [num_asp, emb_size]
        #     span_embs = torch.cat([torch.unsqueeze(dummy_spans, 1), span_embs], dim=1)  # [num_asp, 1 + num_spans, emb_size]
        # else:
        span_embs = torch.cat([torch.unsqueeze(asp_embs, 1), span_embs], dim=1)  # [num_asp, 1 + num_spans, emb_size]
        new_asp_emb = torch.sum(torch.unsqueeze(attn_weights, 2) * span_embs, dim=1)  # [num_asp, emb_size]

        if self.use_gate:
            # get the final embedding after a gate
            gate = self.gate_ffnn(torch.cat([asp_embs, new_asp_emb], dim=1)) # [num_asp, emb_size]
            gate = torch.sigmoid(gate)
            # gate_mean = torch.mean(gate, dim=-1)
            refined_asp_embs = gate * new_asp_emb  + (1-gate) * asp_embs
        else:
            refined_asp_embs = new_asp_emb

        # get the ranking of scores
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True) # 0 denotes the dummy span, and other spans' indexs + 1
        return refined_asp_embs, indices, attn_weights, sorted_scores





    def get_forward_data(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                multi_labels=None, multi_aspects_pos=None, sentiment_scores=None, wps2tokens=None,
                dep_distances=None, multi_bias_labels=None, starts_position=None, ends_position=None,
                senti_distributions=None):
        device = input_ids.device
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=None,  # RoBERTa do not have token_type_ids
                               position_ids=position_ids)
        sequence_output = outputs[0]  # [bs, seq_len, emb_size]

        all_asp_emb = []
        pure_asp_embs = []
        if self.save_spans_info:
            span_start_idx = []
            span_end_idx = []
            span_senti_scores = []
            span_depend_scores = []
            span_atten_weights = []
            span_total_scores = []
        input_masks = attention_mask.to(torch.bool)  # [num seg, num max tokens]
        bs = input_ids.shape[0]

        # Get labels
        labels = []
        bias_labels = []
        kl_loss = 0
        for bs_idx in range(bs):
            # get labels and aspect terms information
            asps_from = []
            asps_to = []
            _j = 0
            cur_aspects_pos = multi_aspects_pos[bs_idx]
            while _j < len(multi_labels[bs_idx]) and multi_labels[bs_idx][_j] != -1:
                labels.append(multi_labels[bs_idx][_j])
                bias_labels.append(multi_bias_labels[bs_idx][_j])
                asps_from.append(cur_aspects_pos[_j * 2] + 1)  # the first token is [CLS]
                asps_to.append(cur_aspects_pos[_j * 2 + 1] + 1)
                _j += 1
            num_asp = _j
            asps_from = torch.tensor(asps_from, dtype=int, device=device)
            asps_to = torch.tensor(asps_to, dtype=int, device=device)

            # get wordpieces embeddings
            input_mask = input_masks[bs_idx]
            valid_sent = sequence_output[bs_idx][input_mask]  # [num_wps, hidden_size]
            num_wps = valid_sent.shape[0] - 2  # the head and tail are [CLS] and [SEG] respectively
            valid_dep_dis = dep_distances[bs_idx][0:num_asp, 0:num_wps]

            # all possible spans' position
            starts = starts_position[bs_idx]
            num_spans = 0
            for _tmp in starts:
                if _tmp == -1:
                    break
                else:
                    num_spans += 1
            starts = starts[:num_spans]
            ends = ends_position[bs_idx][:num_spans]
            senti_classifier_logits = senti_distributions[bs_idx][:num_spans]

            # --------------- Get the embeddings of aspects and all spans ----------------------
            # add aspects info to span's starts and ends list to get span embeddings
            starts, ends = torch.cat((asps_from, starts)), torch.cat((asps_to, ends))

            # get all spans and aspect terms embedding
            start_embs, end_embs = valid_sent[starts], valid_sent[ends - 1]
            span_emb_compo_list = [start_embs, end_embs]
            if self.use_width_features:
                width_idx = ends - starts
                width_idx = torch.clamp(width_idx, 0, self.max_span_width)
                width_emb = self.emb_span_width(width_idx - 1)
                width_emb = self.dropout(width_emb)
                span_emb_compo_list.append(width_emb)

            span_tokens = torch.unsqueeze(torch.arange(1, num_wps + 1, device=device), 0).repeat(num_asp + num_spans,
                                                                                                 1)  # [num_asp+num_spans, num_wps]
            span_tokens_mask = (span_tokens >= torch.unsqueeze(starts, 1)) & (
                    span_tokens < torch.unsqueeze(ends, 1))  # [num_asp+num_spans, num_wps]
            token_attn = torch.squeeze(self.attn_mlp(valid_sent[1:-1]))  # [num_wps]
            span_tokens_attn_raw = torch.log(span_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn,
                                                                                                 0)  # [num_asp+num_spans, num_wps]  (inf & valuse)
            span_tokens_attn = nn.functional.softmax(span_tokens_attn_raw, dim=1)  # [num_asp+num_spans, num_wps]
            span_attn_emb = torch.matmul(span_tokens_attn, valid_sent[1:-1])  # [num_asp+num_spans, emb_size]
            span_emb_compo_list.append(span_attn_emb)
            span_emb = torch.cat(span_emb_compo_list, dim=1)  # [num_asp+num_spans, new_size]
            asp_emb, span_emb = span_emb[:num_asp], span_emb[num_asp:]  # [num_asp, new_size],  [num_spans, new_size]
            pure_asp_embs.append(asp_emb)
            starts, ends = starts[num_asp:], ends[num_asp:]

            # --------------- Get the embeddings of aspects and all spans finished! ----------------------

            span_sentiment_scores = self.span_emb_score_ffnn(span_emb)
            if self.no_distillation:
                kl_loss += torch.tensor(0, device=device)
            else:
                assert span_sentiment_scores.shape == senti_classifier_logits.shape
                span_kl_loss_fn = nn.KLDivLoss(reduction='mean')
                span_kl_loss = span_kl_loss_fn(torch.log_softmax(span_sentiment_scores, dim=-1),
                                               torch.softmax(senti_classifier_logits, dim=-1))
                kl_loss += span_kl_loss

            if self.no_sentiment_score:
                span_sentiment_scores = torch.ones_like(span_sentiment_scores, device=device)[:, 1]
                span_meaningless_scores = span_sentiment_scores
            else:
                span_sentiment_scores = 1 - torch.softmax(span_sentiment_scores, dim=-1)[:, 1]
                span_meaningless_scores = 1 - span_sentiment_scores


            repeat_asp_emb = torch.unsqueeze(asp_emb, 1).repeat(1, num_spans, 1)  # [num_asp, num_spans, new_size]
            similarity_emb = repeat_asp_emb * span_emb  # [num_asp, num_spans, new_size]

            position_info = torch.stack([starts, ends], dim=0).t()  # [num_spans, 2]
            rela_position = torch.cat([position_info for _ in range(num_asp)], dim=1)  # [num_spans, num_asp*2]
            cur_aspects_pos = torch.unsqueeze(cur_aspects_pos[:num_asp * 2], dim=0).repeat(num_spans,
                                                                                           1)  # [num_spans, num_asp*2]
            cur_aspects_pos = abs(cur_aspects_pos - rela_position)  # [num_spans, num_asp*2]
            rela_position = []
            for _n in range(0, num_asp * 2, 2):
                rela_position.append(torch.min(cur_aspects_pos[:, _n], cur_aspects_pos[:, _n + 1]))
            rela_position = torch.stack(rela_position, dim=1)  # [num_spans, num_asp]
            rela_position = torch.clamp(rela_position, 0, self.max_rela_distance - 1)
            if self.use_rela_dis_features:
                rela_position_emb = self.emb_rela_distance(rela_position)
                rela_position_emb = self.dropout(rela_position_emb).permute(1, 0,
                                                                            2)  # [num_asp, num_spans, feature_size]
                pair_emb = torch.cat(
                    [repeat_asp_emb, torch.unsqueeze(span_emb, 0).repeat(num_asp, 1, 1), similarity_emb,
                     rela_position_emb], dim=-1)  # [num_asp, num_spans, pair_size]
            elif self.use_dep_dis_features:
                assert valid_dep_dis.shape == (num_asp, num_wps)
                starts_idx = (starts - 1).unsqueeze(dim=0).repeat(num_asp, 1) # [num_asp, nums_span]
                ends_idx = (ends - 1).unsqueeze(dim=0).repeat(num_asp, 1)
                span_dep_distances_start = valid_dep_dis.gather(dim=-1, index=starts_idx)
                span_dep_distances_end = valid_dep_dis.gather(dim=-1, index=ends_idx - 1)
                span_dep_distances = torch.empty(num_asp, num_spans, device=device, dtype=torch.long)
                torch.min(span_dep_distances_start, span_dep_distances_end, out=span_dep_distances)
                span_dep_distances = torch.clamp(span_dep_distances, 0, self.max_dep_dis_distance - 1)
                dep_dis_position_emb = self.emb_dep_dis_distance(span_dep_distances)
                dep_dis_position_emb = self.dropout(dep_dis_position_emb)  # [num_asp, num_spans, feature_size]
                pair_emb = torch.cat(
                    [repeat_asp_emb, torch.unsqueeze(span_emb, 0).repeat(num_asp, 1, 1), similarity_emb,
                     dep_dis_position_emb], dim=-1)  # [num_asp, num_spans, pair_size]

            else:
                pair_emb = torch.cat(
                    [repeat_asp_emb, torch.unsqueeze(span_emb, 0).repeat(num_asp, 1, 1), similarity_emb],
                    dim=-1)  # [num_asp, num_spans, pair_size]
            dependency_scores = torch.softmax(torch.squeeze(self.dependency_score_ffnn(pair_emb), 2), dim=-1)  # [num_asp, num_spans], remove torch.tank
            scores =  span_sentiment_scores * dependency_scores
            dummy_scores = torch.unsqueeze(torch.mean(span_meaningless_scores * dependency_scores, dim=-1), dim=1) # num_asp
            assert len(dummy_scores) == num_asp


            # refine_aspect_with_before_dummy
            weighted_asp_emb, cur_indice, atten_weights, sorted_scores = self.refine_aspect(asp_emb, span_emb, scores, dummy_scores=dummy_scores)
            all_asp_emb.append(weighted_asp_emb)

            # get the top-N spans' information
            if self.save_spans_info:
                topn = self.top_n
                indices = cur_indice[:, 0:topn]  # [num_asp * top_n]
                if num_spans < topn:
                    indices = torch.cat([indices, indices.repeat(1, topn)], dim=-1)[:, 0:topn]

                starts = torch.cat([torch.tensor([0], device=device), starts],
                                   dim=0) - 1  # [1 + num_spans], -1 denotes the dummy span. others denote the position in sentence text
                ends = torch.cat([torch.tensor([0], device=device), ends], dim=0) - 1
                span_start_idx.append(starts[indices])
                span_end_idx.append(ends[indices])
                span_sentiment_scores = torch.cat([torch.tensor([0], device=device), span_sentiment_scores],
                                                  dim=0)  # [1 + num_spans],
                span_senti_scores.append(span_sentiment_scores[indices])
                dependency_scores = torch.cat(
                    [torch.zeros(num_asp, 1, dtype=torch.float32, device=device), dependency_scores],
                    dim=1)  # [num_asp. 1 + num_spans],
                span_depend_scores.append(dependency_scores.gather(dim=-1, index=indices))
                span_atten_weights.append(atten_weights.gather(dim=-1, index=indices))

                scores = torch.cat(
                    [dummy_scores, scores],
                    dim=1)
                span_total_scores.append(scores.gather(dim=-1, index=indices))

        labels = torch.tensor(labels, dtype=torch.long, device=input_ids.device)
        aspects_emb = torch.cat(all_asp_emb, dim=0)
        kl_loss = kl_loss / bs

        bias_labels = torch.tensor(bias_labels, dtype=torch.long, device=input_ids.device)
        pure_asp_embs = torch.cat(pure_asp_embs, dim=0)


        if self.save_spans_info:
            span_start_idx = torch.cat(span_start_idx, dim=0)
            span_end_idx = torch.cat(span_end_idx, dim=0)
            span_atten_weights = torch.cat(span_atten_weights, dim=0)
            span_senti_scores = torch.cat(span_senti_scores, dim=0)
            span_depend_scores = torch.cat(span_depend_scores, dim=0)
            span_total_scores = torch.cat(span_total_scores, dim = 0)

            outputs = [aspects_emb, labels, pure_asp_embs, bias_labels, kl_loss,
                       span_start_idx, span_end_idx, span_atten_weights,
                       span_total_scores, span_senti_scores, span_depend_scores]
        else:
            outputs = [aspects_emb, labels, pure_asp_embs, bias_labels, kl_loss]


        return outputs

    def encoder_forward(self, aspects_emb, labels, pure_asp_embs, kl_loss, bias_labels):

        # absa classification loss
        logits = self.classifier(aspects_emb)
        cls_loss_fn = nn.CrossEntropyLoss()
        cls_loss = cls_loss_fn(logits,labels)

        # adversarial classification loss
        adv_logits = self.adv_classifier(pure_asp_embs)
        adv_loss_fn = nn.CrossEntropyLoss()
        adv_labels = torch.ones(labels.shape, dtype=torch.long, device=labels.device)

        if self.no_adversarial or self.no_adv_loss:
            adv_loss = torch.tensor([0.0], device=labels.device)
        elif self.adversarial_on_polarity:
            mask = (bias_labels != 1)
            tmp_bias_labels = bias_labels[mask]
            tmp_adv_logits = adv_logits[mask]
            tmp_adv_labels = torch.ones(tmp_bias_labels.shape, dtype=torch.long, device=labels.device)
            adv_loss = adv_loss_fn(tmp_adv_logits, tmp_adv_labels)

        else:
            adv_loss = adv_loss_fn(adv_logits, adv_labels)




        cls_and_adv_loss = cls_loss + self.adv_loss_weight * adv_loss + kl_loss

        outputs = [cls_and_adv_loss, logits, labels, cls_loss, adv_loss, kl_loss]

        return tuple(outputs)

    def discriminator_forward(self, pure_asp_embs, bias_labels):

        pure_logits = self.discriminator(pure_asp_embs)
        disc_loss_fn = nn.CrossEntropyLoss()
        if self.no_adversarial:
            disc_loss = torch.tensor([0.0], device=pure_asp_embs.device)
        elif self.adversarial_on_polarity:
            mask = (bias_labels != 1)
            tmp_pure_logits = pure_logits[mask]
            tmp_bias_labels = bias_labels[mask]
            disc_loss = disc_loss_fn(tmp_pure_logits, tmp_bias_labels)
        else:
            disc_loss = disc_loss_fn(pure_logits, bias_labels)

        outputs = [disc_loss, pure_logits]

        return tuple(outputs)

class BERTSpanCLSAdvDistillation(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTSpanCLSAdvDistillation, self).__init__(config)
        self.num_labels = self.config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.senti_classifier = RoBERTaSpanCLS.from_pretrained(config.phrase_sentiment_model_path, config=config)


        self.attn_mlp = nn.Linear(config.hidden_size, 1)

        # self.hidden_size = config.hidden_size

        self.save_spans_info = config.save_spans_info
        self.use_gate = config.use_gate
        self.no_distillation = config.no_distillation
        self.no_sentiment_score = config.no_sentiment_score
        self.top_n = config.top_n  # top_n spans according to the total score
        self.max_span_width = config.max_span_width
        self.use_width_features = config.use_width_features
        self.use_rela_dis_features = config.use_rela_dis_features
        self.use_dep_dis_features = config.use_dep_dis_features
        self.feature_emb_size = config.feature_emb_size
        self.span_emb_size = config.hidden_size * 3 + self.feature_emb_size if self.use_width_features else config.hidden_size * 3
        self.max_rela_distance = config.max_rela_distance
        self.adv_loss_weight = config.adv_loss_weight
        self.max_dep_dis_distance = 20
        self.pair_emb_size = self.span_emb_size * 3  + self.feature_emb_size if (self.use_rela_dis_features or self.use_dep_dis_features) else self.span_emb_size * 3

        self.absa_cls = self.make_ffnn(self.span_emb_size, self.span_emb_size, self.num_labels)

        self.emb_span_width = nn.Embedding(self.max_span_width, self.feature_emb_size) if self.use_width_features else None
        self.emb_rela_distance = nn.Embedding(self.max_rela_distance, self.feature_emb_size) if self.use_rela_dis_features else None
        self.emb_dep_dis_distance = nn.Embedding(self.max_dep_dis_distance, self.feature_emb_size) if self.use_dep_dis_features else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, 1000, output_size=3)
        self.dependency_score_ffnn = self.make_ffnn(self.pair_emb_size, 1000, output_size=1)
        self.gate_ffnn = self.make_ffnn(self.span_emb_size * 2, 1000, output_size=self.span_emb_size)

        config.span_emb_size = self.span_emb_size

        # discriminator: to classify the polarity of aspect
        # single dense layer
        self.nums_bias_label = config.nums_bias_label
        # self.disc = nn.Linear(self.span_emb_size, self.nums_bias_label)
        # self.adv_cls = nn.Linear(self.span_emb_size, self.nums_bias_label)

        # two-layer MLP

        self.disc = self.make_ffnn(self.span_emb_size, self.span_emb_size, self.nums_bias_label)

    def get_params(self, named=True):
        """
        Returns:
            disc_params: parameters of the discriminator/adversary
            other_params       : parameters of the roberta, classifiers and other remain parts
        """
        # encoder parameters
        encoder_params = []
        for name, param in self.named_parameters():
            if name.startswith('bert') or name.startswith('roberta'):
                to_add = (name, param) if named else param
                encoder_params.append(to_add)

        # task-based parameters
        task_params = list(list(self.absa_cls.parameters()) + \
            list(self.attn_mlp.parameters()) + list(self.span_emb_score_ffnn.parameters()) + \
            list(self.dependency_score_ffnn.parameters()) + list(self.gate_ffnn.parameters()))
        if self.use_width_features:
            task_params += list(self.emb_span_width.parameters())
        if self.use_rela_dis_features:
            task_params += list(self.emb_rela_distance.parameters())
        if self.use_dep_dis_features:
            task_params += list(self.emb_dep_dis_distance.parameters())

        # discriminator parameters
        disc_params = self.disc.parameters()

        return encoder_params, task_params, disc_params

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        return nn.Linear(in_features, out_features, bias)

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def classifier(self, emb):
        cls_feature = self.dropout(emb)
        logits = self.absa_cls(cls_feature)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def discriminator(self, emb):
        # Note: detach the embedding since when don't want the gradient to flow
        #       all the way to the encoder. disc_loss is used only to change the
        #       parameters of the discriminator network
        pure_logits = self.disc(self.dropout(emb.detach()))
        return pure_logits

    def adv_classifier(self, emb):
        # Note: detach the embedding since when don't want the gradient to flow
        #       all the way to the encoder. disc_loss is used only to change the
        #       parameters of the discriminator network
        pure_logits = self.disc(self.dropout(emb))
        return pure_logits

    def refine_aspect(self, asp_embs, span_embs, scores, max_num=None, dummy_scores=None):
        # asp_embs: [num_asp, emb_size]
        # span_embs: [num_spans, emb_size]
        # scores: [num_asp, num_spans]

        # add dummy scores and get attention weights
        num_asp = asp_embs.shape[0]
        # dummy_scores = torch.full((num_asp, 1), fill_value=9,device=scores.device) # the dummy span's score
        # scores = torch.cat([dummy_scores, scores], dim=1)
        if dummy_scores is not None:
            scores = torch.cat([dummy_scores, scores], dim=1)
        else:
            scores = torch.cat([torch.zeros(num_asp, 1, device=scores.device), scores], dim=1) # [num_asp, 1 + num_spans]
        attn_weights = nn.functional.softmax(scores, dim=-1) # [num_asp, 1 + num_spans]

        # get the new asp_embs with attention weight
        span_embs = torch.unsqueeze(span_embs, 0).repeat(num_asp, 1, 1)  # [num_asp, num_spans, emb_size]
        span_embs = torch.cat([torch.unsqueeze(asp_embs, 1), span_embs], dim=1)  # [num_asp, 1 + num_spans, emb_size]
        new_asp_emb = torch.sum(torch.unsqueeze(attn_weights, 2) * span_embs, dim=1)  # [num_asp, emb_size]

        if self.use_gate:
            # get the final embedding after a gate
            gate = self.gate_ffnn(torch.cat([asp_embs, new_asp_emb], dim=1)) # [num_asp, emb_size]
            gate = torch.sigmoid(gate)
            gate_mean = torch.mean(gate, dim=-1)
            refined_asp_embs = gate * new_asp_emb  + (1-gate) * asp_embs
        else:
            refined_asp_embs = new_asp_emb

        # get the ranking of scores
        sorted_scores, indices = torch.sort(scores, dim=1, descending=True) # 0 denotes the dummy span, and other spans' indexs + 1
        return refined_asp_embs, indices, attn_weights, sorted_scores

    def get_forward_data(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                multi_labels=None, multi_aspects_pos=None, sentiment_scores=None, wps2tokens=None,
                dep_distances=None, multi_bias_labels=None, starts_position=None, ends_position=None,
                senti_distributions=None):
        device = input_ids.device
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,  # RoBERTa do not have token_type_ids
                               position_ids=position_ids)
        sequence_output = outputs[0]  # [bs, seq_len, emb_size]

        all_asp_emb = []
        pure_asp_embs = []
        if self.save_spans_info:
            span_start_idx = []
            span_end_idx = []
            span_senti_scores = []
            span_depend_scores = []
            span_atten_weights = []
            span_total_scores = []
        input_masks = attention_mask.to(torch.bool)  # [num seg, num max tokens]
        bs = input_ids.shape[0]

        # Get labels
        labels = []
        bias_labels = []
        kl_loss = 0
        for bs_idx in range(bs):
            # get labels and aspect terms information
            asps_from = []
            asps_to = []
            _j = 0
            cur_aspects_pos = multi_aspects_pos[bs_idx]
            while _j < len(multi_labels[bs_idx]) and multi_labels[bs_idx][_j] != -1:
                labels.append(multi_labels[bs_idx][_j])
                bias_labels.append(multi_bias_labels[bs_idx][_j])
                asps_from.append(cur_aspects_pos[_j * 2] + 1)  # the first token is [CLS]
                asps_to.append(cur_aspects_pos[_j * 2 + 1] + 1)
                _j += 1
            num_asp = _j
            asps_from = torch.tensor(asps_from, dtype=int, device=device)
            asps_to = torch.tensor(asps_to, dtype=int, device=device)

            # get wordpieces embeddings
            input_mask = input_masks[bs_idx]
            valid_sent = sequence_output[bs_idx][input_mask]  # [num_wps, hidden_size]
            num_wps = valid_sent.shape[0] - 2  # the head and tail are [CLS] and [SEG] respectively
            valid_dep_dis = dep_distances[bs_idx][0:num_asp, 0:num_wps]

            # all possible spans' position
            starts = starts_position[bs_idx]
            num_spans = 0
            for _tmp in starts:
                if _tmp == -1:
                    break
                else:
                    num_spans += 1
            starts = starts[:num_spans]
            ends = ends_position[bs_idx][:num_spans]
            senti_classifier_logits = senti_distributions[bs_idx][:num_spans]



            # --------------- Get the embeddings of aspects and all spans ----------------------
            # add aspects info to span's starts and ends list to get span embeddings
            starts, ends = torch.cat((asps_from, starts)), torch.cat((asps_to, ends))

            # get all spans and aspect terms embedding
            start_embs, end_embs = valid_sent[starts], valid_sent[ends - 1]
            span_emb_compo_list = [start_embs, end_embs]
            if self.use_width_features:
                width_idx = ends - starts
                width_idx = torch.clamp(width_idx, 0, self.max_span_width)
                width_emb = self.emb_span_width(width_idx - 1)
                width_emb = self.dropout(width_emb)
                span_emb_compo_list.append(width_emb)

            span_tokens = torch.unsqueeze(torch.arange(1, num_wps + 1, device=device), 0).repeat(num_asp + num_spans,
                                                                                                 1)  # [num_asp+num_spans, num_wps]
            span_tokens_mask = (span_tokens >= torch.unsqueeze(starts, 1)) & (
                    span_tokens < torch.unsqueeze(ends, 1))  # [num_asp+num_spans, num_wps]
            token_attn = torch.squeeze(self.attn_mlp(valid_sent[1:-1]))  # [num_wps]
            span_tokens_attn_raw = torch.log(span_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn,
                                                                                                 0)  # [num_asp+num_spans, num_wps]  (inf & valuse)
            span_tokens_attn = nn.functional.softmax(span_tokens_attn_raw, dim=1)  # [num_asp+num_spans, num_wps]
            span_attn_emb = torch.matmul(span_tokens_attn, valid_sent[1:-1])  # [num_asp+num_spans, emb_size]
            span_emb_compo_list.append(span_attn_emb)
            span_emb = torch.cat(span_emb_compo_list, dim=1)  # [num_asp+num_spans, new_size]
            # span_emb = self.dense(span_emb)  # TODO: note
            asp_emb, span_emb = span_emb[:num_asp], span_emb[num_asp:]  # [num_asp, new_size],  [num_spans, new_size]
            pure_asp_embs.append(asp_emb)
            starts, ends = starts[num_asp:], ends[num_asp:]

            # --------------- Get the embeddings of aspects and all spans finished! ----------------------

            span_sentiment_scores = self.span_emb_score_ffnn(span_emb)
            if self.no_distillation:
                kl_loss += torch.tensor(0, device=device)
            else:
                assert span_sentiment_scores.shape == senti_classifier_logits.shape
                span_kl_loss_fn = nn.KLDivLoss(reduction='mean')
                span_kl_loss = span_kl_loss_fn(torch.log_softmax(span_sentiment_scores, dim=-1),
                                               torch.softmax(senti_classifier_logits, dim=-1))
                kl_loss += span_kl_loss

            if self.no_sentiment_score:
                span_sentiment_scores = torch.ones_like(span_sentiment_scores, device=device)[:, 1]
                span_meaningless_scores = span_sentiment_scores
            else:
                span_sentiment_scores = 1 - torch.softmax(span_sentiment_scores, dim=-1)[:, 1]
                span_meaningless_scores = 1 - span_sentiment_scores


            repeat_asp_emb = torch.unsqueeze(asp_emb, 1).repeat(1, num_spans, 1)  # [num_asp, num_spans, new_size]
            similarity_emb = repeat_asp_emb * span_emb  # [num_asp, num_spans, new_size]

            position_info = torch.stack([starts, ends], dim=0).t()  # [num_spans, 2]
            rela_position = torch.cat([position_info for _ in range(num_asp)], dim=1)  # [num_spans, num_asp*2]
            cur_aspects_pos = torch.unsqueeze(cur_aspects_pos[:num_asp * 2], dim=0).repeat(num_spans,
                                                                                           1)  # [num_spans, num_asp*2]
            cur_aspects_pos = abs(cur_aspects_pos - rela_position)  # [num_spans, num_asp*2]
            rela_position = []
            for _n in range(0, num_asp * 2, 2):
                rela_position.append(torch.min(cur_aspects_pos[:, _n], cur_aspects_pos[:, _n + 1]))
            rela_position = torch.stack(rela_position, dim=1)  # [num_spans, num_asp]
            rela_position = torch.clamp(rela_position, 0, self.max_rela_distance - 1)
            if self.use_rela_dis_features:
                rela_position_emb = self.emb_rela_distance(rela_position)
                rela_position_emb = self.dropout(rela_position_emb).permute(1, 0,
                                                                            2)  # [num_asp, num_spans, feature_size]
                pair_emb = torch.cat(
                    [repeat_asp_emb, torch.unsqueeze(span_emb, 0).repeat(num_asp, 1, 1), similarity_emb,
                     rela_position_emb], dim=-1)  # [num_asp, num_spans, pair_size]
            elif self.use_dep_dis_features:
                assert valid_dep_dis.shape == (num_asp, num_wps)
                # span_dep_distances = [[] for _ in range(num_asp)]
                # for _asp_idx in range(0, num_asp):
                #     for _span_idx in range(num_spans):
                #         cur_min_dis = min(valid_dep_dis[_asp_idx][(starts[num_asp:][_span_idx]-1):(ends[num_asp:][_span_idx]-1)])
                #         span_dep_distances[_asp_idx].append(cur_min_dis)
                starts_idx = (starts - 1).unsqueeze(dim=0).repeat(num_asp, 1) # [num_asp, nums_span]
                ends_idx = (ends - 1).unsqueeze(dim=0).repeat(num_asp, 1)
                span_dep_distances_start = valid_dep_dis.gather(dim=-1, index=starts_idx)
                span_dep_distances_end = valid_dep_dis.gather(dim=-1, index=ends_idx - 1)
                span_dep_distances = torch.empty(num_asp, num_spans, device=device, dtype=torch.long)
                # TODO: get the average value among the complete span rather than the start or end wordpiece
                torch.min(span_dep_distances_start, span_dep_distances_end, out=span_dep_distances)
                # span_dep_distances = torch.tensor(span_dep_distances, device=device)
                span_dep_distances = torch.clamp(span_dep_distances, 0, self.max_dep_dis_distance - 1)
                dep_dis_position_emb = self.emb_dep_dis_distance(span_dep_distances)
                dep_dis_position_emb = self.dropout(dep_dis_position_emb)  # [num_asp, num_spans, feature_size]
                pair_emb = torch.cat(
                    [repeat_asp_emb, torch.unsqueeze(span_emb, 0).repeat(num_asp, 1, 1), similarity_emb,
                     dep_dis_position_emb], dim=-1)  # [num_asp, num_spans, pair_size]
                # pair_emb = torch.cat(
                #     [similarity_emb, repeat_asp_emb - torch.unsqueeze(span_emb, 0).repeat(num_asp, 1, 1),
                #      dep_dis_position_emb], dim=-1)  # [num_asp, num_spans, pair_size]

            else:
                pair_emb = torch.cat(
                    [repeat_asp_emb, torch.unsqueeze(span_emb, 0).repeat(num_asp, 1, 1), similarity_emb],
                    dim=-1)  # [num_asp, num_spans, pair_size]

            # TODO: alter the ways of getting dependency scores and make sure the score is greater than 0
            dependency_scores = torch.softmax(torch.squeeze(self.dependency_score_ffnn(pair_emb), 2), dim=-1)  # [num_asp, num_spans], remove torch.tank
            # dependency_scores = self._rescal(dependency_scores, 0.01, 20)
            scores =  span_sentiment_scores * dependency_scores
            dummy_scores = torch.unsqueeze(torch.mean(span_meaningless_scores * dependency_scores, dim=-1), dim=1) # num_asp
            assert len(dummy_scores) == num_asp
            # scores = dependency_scores
            # scores = span_sentiment_scores.squeeze(0).repeat(num_asp, 1)


            # refine_aspect_with_before_dummy
            weighted_asp_emb, cur_indice, atten_weights, sorted_scores = self.refine_aspect(asp_emb, span_emb, scores, dummy_scores=dummy_scores)
            all_asp_emb.append(weighted_asp_emb)

            # get the top-N spans' information
            if self.save_spans_info:
                topn = self.top_n
                indices = cur_indice[:, 0:topn]  # [num_asp * top_n]
                if num_spans < topn:
                    indices = torch.cat([indices, indices.repeat(1, topn)], dim=-1)[:, 0:topn]

                starts = torch.cat([torch.tensor([0], device=device), starts],
                                   dim=0) - 1  # [1 + num_spans], -1 denotes the dummy span. others denote the position in sentence text
                ends = torch.cat([torch.tensor([0], device=device), ends], dim=0) - 1
                span_start_idx.append(starts[indices])
                span_end_idx.append(ends[indices])
                span_sentiment_scores = torch.cat([torch.tensor([0], device=device), span_sentiment_scores],
                                                  dim=0)  # [1 + num_spans],
                span_senti_scores.append(span_sentiment_scores[indices])
                dependency_scores = torch.cat(
                    [torch.zeros(num_asp, 1, dtype=torch.float32, device=device), dependency_scores],
                    dim=1)  # [num_asp. 1 + num_spans],
                span_depend_scores.append(dependency_scores.gather(dim=-1, index=indices))
                span_atten_weights.append(atten_weights.gather(dim=-1, index=indices))
                # scores = torch.cat(
                #     [torch.zeros(num_asp, 1, dtype=torch.float32, device=device), scores],
                #     dim=1)
                scores = torch.cat(
                    [dummy_scores, scores],
                    dim=1)
                span_total_scores.append(scores.gather(dim=-1, index=indices))

        labels = torch.tensor(labels, dtype=torch.long, device=input_ids.device)
        aspects_emb = torch.cat(all_asp_emb, dim=0)
        kl_loss = kl_loss / bs

        bias_labels = torch.tensor(bias_labels, dtype=torch.long, device=input_ids.device)
        pure_asp_embs = torch.cat(pure_asp_embs, dim=0)

        bias_labels = labels

        if self.save_spans_info:
            span_start_idx = torch.cat(span_start_idx, dim=0)
            span_end_idx = torch.cat(span_end_idx, dim=0)
            span_atten_weights = torch.cat(span_atten_weights, dim=0)
            span_senti_scores = torch.cat(span_senti_scores, dim=0)
            span_depend_scores = torch.cat(span_depend_scores, dim=0)
            # span_total_scores = span_senti_scores + span_depend_scores
            span_total_scores = torch.cat(span_total_scores, dim = 0)

            outputs = [aspects_emb, labels, pure_asp_embs, bias_labels, kl_loss,
                       span_start_idx, span_end_idx, span_atten_weights,
                       span_total_scores, span_senti_scores, span_depend_scores]
        else:
            outputs = [aspects_emb, labels, pure_asp_embs, bias_labels, kl_loss]

        # view the gate value
        # gate_value = self.gate_ffnn.parameters()
        # print(gate_value)
        # gate_value_list = [params.data for params in list(gate_value)]
        # for i, params in enumerate(gate_value_list):
        #     if i % 2 == 0:
        #         print(torch.mean(torch.mean(params, dim=-1), dim=-1))
        #     else:
        #         print(torch.mean(params, dim=-1))


        return outputs

    def encoder_forward(self, aspects_emb, labels, pure_asp_embs, kl_loss):

        # absa classification loss
        logits = self.classifier(aspects_emb)
        cls_loss_fn = nn.CrossEntropyLoss()
        cls_loss = cls_loss_fn(logits,labels)

        # adversarial classification loss
        adv_logits = self.adv_classifier(pure_asp_embs)
        adv_loss_fn = nn.CrossEntropyLoss()
        adv_labels = torch.ones(labels.shape, dtype=torch.long, device=labels.device)
        adv_loss = adv_loss_fn(adv_logits, adv_labels)

        cls_and_adv_loss = cls_loss + self.adv_loss_weight * adv_loss + kl_loss

        outputs = [cls_and_adv_loss, logits, labels, cls_loss, adv_loss, kl_loss]

        return tuple(outputs)

    def discriminator_forward(self, pure_asp_embs, bias_labels):
        pure_logits = self.discriminator(pure_asp_embs)
        disc_loss_fn = nn.CrossEntropyLoss()
        disc_loss = disc_loss_fn(pure_logits, bias_labels)

        outputs = [disc_loss, pure_logits]

        return tuple(outputs)





