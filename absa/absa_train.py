import argparse
from tqdm import tqdm, trange
import pandas as pd
from peach.help import *

from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from absa.absa_dataset import AbsaDataset, get_raw_datasets, DATASET2PATH
from absa.models import BertMultiAspectsCLS, RoBERTaMultiAspectsCLS, \
    RoBERTaSpanCLS, RoBERTaSpanCLSAdvDistillation, BERTSpanCLSAdvDistillation
from peach.utils_tokenizer import split_to_token, parse_tk_idx_list_wrt_char, continue_tokenize_for_wordpiece, char_to_token_span
import json
from random import random

def get_wordnet_bias(args, raw_dataset, dataset_name, dataset_type, see_bias=False):
    with open(os.path.join("./datasets/SentiWords.json")) as f:
        senti_dict = json.load(f)

    for _idx, sent_dict in enumerate(raw_dataset):
        for term_dict in sent_dict["terms"]:
            if term_dict["polarity"] == "conflict":
                continue
            asp_term_list = term_dict["term"].split(' ')
            tmp_score = 0
            for asp_term in asp_term_list:
                if asp_term in senti_dict:
                    tmp_score += senti_dict[asp_term]
            if tmp_score > 0:
                term_dict['bias'] = "positive"
            elif tmp_score == 0:
                term_dict['bias'] = "neutral"
            elif tmp_score < 0:
                term_dict['bias'] = "negative"

    if dataset_name in ['rest14', 'lap14']:
        save_path = "./datasets/SemEval-14/bias_" + dataset_name + "/wordnet_bias_"+ dataset_type + ".json"
    elif dataset_name == 'twitter':
        save_path = "./datasets/Twitter/bias_" + dataset_name + "/wordnet_bias_" + dataset_type + ".json"
    elif dataset_name == 'rest15':
        save_path = "./datasets/SemEval-15/bias_" + dataset_name + "/wordnet_bias_" + dataset_type + ".json"
    elif dataset_name == 'rest16':
        save_path = "./datasets/SemEval-16/bias_" + dataset_name + "/wordnet_bias_" + dataset_type + ".json"
    with open(save_path, 'w') as f_obj:
        json.dump(raw_dataset, f_obj)

    if see_bias:
        sent_text_list = []
        aspect_list = []
        label_list = []
        bias_list = []
        for _idx, sent_dict in enumerate(raw_dataset.absa_dataset):
            sent_text = sent_dict["sent_text"]
            for term_dict in sent_dict["terms"]:
                if term_dict["polarity"] == "conflict":
                    continue
                sent_text_list.append(sent_text)
                aspect_list.append(term_dict["term"])
                label_list.append(term_dict["polarity"])
                bias_list.append(term_dict["bias"])

        pred_data = {
            "sent_text": sent_text_list,
            "aspect": aspect_list,
            "label": label_list,
            "bias": bias_list,
        }
        pred_data_df = pd.DataFrame(pred_data)
        pred_data_df.to_csv(os.path.join(args.output_dir, "./view_wordnet_bias_"+ dataset_type + ".csv"))

def get_distillation_dataset(args, config, tokenizer, dataset_name):
    senti_classifier = RoBERTaSpanCLS.from_pretrained(args.phrase_sentiment_model_path,
                                                           config=config)
    senti_classifier.to(args.device)

    sep_token, cls_token, pad_token = \
        tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token
    sep_id, cls_id, pad_id = tokenizer.convert_tokens_to_ids(
        [sep_token, cls_token, pad_token],
    )

    train_dataset_raw, dev_dataset_raw, test_dataset_raw = get_raw_datasets(dataset_name + "_wordnet_bias")
    save_path = DATASET2PATH[dataset_name + "_wordnet_bias"]
    for idx, absa_dataset in enumerate([train_dataset_raw, test_dataset_raw]):
        type = "train" if idx == 0 else "test"

        for _ex_idx, example in enumerate(tqdm(absa_dataset)):
            sent_text = example["sent_text"]
            example["sent_text"] = sent_text.replace('’', '\'')  # '’' will make a difficult for the alignment between tokenizer and parser
            sent_text = example["sent_text"]
            sent_tokens = split_to_token(tokenizer, sent_text)
            charidx2tokenidx = parse_tk_idx_list_wrt_char(tokenizer, sent_text, sent_tokens)
            wp_list, id_list, wpidx2tokenidx = continue_tokenize_for_wordpiece(tokenizer, sent_tokens)
            aspects_pos = []
            term_mask = [0] * len(wpidx2tokenidx)
            for term_dict in example["terms"]:
                from_idx = term_dict["from"]
                to_idx = term_dict["to"]
                token_from_idx, token_to_idx = char_to_token_span(charidx2tokenidx, from_idx, to_idx)
                wp_from_idx = wpidx2tokenidx.index(token_from_idx)
                if token_to_idx > wpidx2tokenidx[-1]:
                    wp_to_idx = len(wpidx2tokenidx)
                else:
                    wp_to_idx = wpidx2tokenidx.index(token_to_idx)
                aspects_pos.extend([wp_from_idx, wp_to_idx])  # save the aspects' position
                for pos_idx in range(wp_from_idx, wp_to_idx):
                    term_mask[pos_idx] = 1

            input_ids = [cls_id] + id_list + [sep_id]

            # provide the sentiment score and wpidx2tokenidx list in train phrase;
            wpidx2tokenidx = torch.tensor(wpidx2tokenidx, dtype=torch.long)

            # prepare for providing wpidx2tokenidx
            wpidx2tokenidx = torch.cat([torch.tensor([-1]), wpidx2tokenidx, torch.tensor([-1])], dim=0) + 1
            assert len(input_ids) == len(wpidx2tokenidx)

            # get the dependency distance between each wordpiece and each aspect, then return a [num_asp, num_wps] tensor

            input_ids = torch.tensor(input_ids, dtype=torch.long)

            # Get all possible spans' position.
            num_wps = len(input_ids) - 2
            num_asp = len(example["terms"])
            starts = torch.unsqueeze(torch.arange(1, num_wps + 1), 1).repeat(1, args.max_span_width)  # [num_wps, max_span_width]
            ends = starts + torch.arange(0, args.max_span_width) + 1  # add 1 to make a slice containing at least one wordpiece
            mask = (ends < (num_wps + 2))
            asps_from = [aspects_pos[_i] + 1 for _i in range(0, num_asp * 2, 2)]  # The first token is [CLS]
            asps_to = [aspects_pos[_i] + 1 for _i in range(1, num_asp * 2, 2)]
            assert len(asps_from) == len(asps_to)

            # Mask the spans do not contain aspect terms
            for _i in range(len(mask)):  # num_wps
                cur_mask = mask[_i]  # [max_span_width]
                for _asp_idx in range(0, num_asp):
                    tmp_mask = (starts[_i] >= asps_to[_asp_idx]) | (ends[_i] <= asps_from[_asp_idx])
                    cur_mask = cur_mask & tmp_mask
                mask[_i] = cur_mask
            starts, ends = starts[mask], ends[mask]

            # Make sure the span contain complete token rather than wordPiece
            cur_wps2tokens = wpidx2tokenidx
            mask = (cur_wps2tokens[starts - 1] != cur_wps2tokens[starts]) & (
                    cur_wps2tokens[ends - 1] != cur_wps2tokens[ends])
            starts, ends = starts[mask], ends[mask]

            # Get all possible spans' position finished.

            # get all possible candidate spans inputs for sentiment classifier
            num_spans = len(starts)
            senti_model_input_ids_list = []
            senti_model_segment_ids_list = []
            senti_model_mask_ids_list = []

            max_span_len = 0
            for _i in range(num_spans):
                cur_senti_model_input_ids = torch.cat(
                    [torch.tensor([input_ids[0]], dtype=torch.long), input_ids[starts[_i]:ends[_i]],
                     torch.tensor([input_ids[-1]], dtype=torch.long)])
                cur_senti_model_segment_ids = torch.tensor([0] * len(cur_senti_model_input_ids), dtype=torch.long)
                cur_senti_model_mask_ids = torch.tensor([1] * len(cur_senti_model_input_ids), dtype=torch.long)
                max_span_len = max(max_span_len, len(cur_senti_model_input_ids))

                senti_model_input_ids_list.append(cur_senti_model_input_ids)
                senti_model_segment_ids_list.append(cur_senti_model_segment_ids)
                senti_model_mask_ids_list.append(cur_senti_model_mask_ids)

            for _i in range(num_spans):
                if len(senti_model_input_ids_list[_i]) < max_span_len:
                    padding_len = max_span_len - len(senti_model_input_ids_list[_i])
                    senti_model_input_ids_list[_i] = torch.cat(
                        [senti_model_input_ids_list[_i],
                         torch.tensor([int(pad_id)] * padding_len, dtype=torch.long)], dim=0)
                    senti_model_mask_ids_list[_i] = torch.cat(
                        [senti_model_mask_ids_list[_i], torch.tensor([0] * padding_len, dtype=torch.long)], dim=0)
                    senti_model_segment_ids_list[_i] = torch.cat(
                        [senti_model_segment_ids_list[_i], torch.tensor([0] * padding_len, dtype=torch.long)],
                        dim=0)

            senti_model_input_ids = torch.stack(senti_model_input_ids_list, dim=0).to(args.device)
            senti_model_mask_ids = torch.stack(senti_model_mask_ids_list, dim=0).to(args.device)

            senti_classifier_logits_list = []
            with torch.no_grad():
                _mini_batch_size = 128
                for _mini_batch_start in range(0, num_spans, _mini_batch_size):
                    if num_spans < _mini_batch_start + _mini_batch_size:
                        _mini_batch_end = num_spans
                    else:
                        _mini_batch_end = _mini_batch_start + _mini_batch_size
                    senti_input_ids = senti_model_input_ids[_mini_batch_start: _mini_batch_end]
                    senti_mask_ids = senti_model_mask_ids[_mini_batch_start: _mini_batch_end]
                    senti_classifier_logits = senti_classifier(senti_input_ids,
                                                                    attention_mask=senti_mask_ids,
                                                                    token_type_ids=None)[
                        0]  # RoBERTa do not have token_type_ids)
                    senti_classifier_logits_list.append(senti_classifier_logits)
            senti_classifier_logits = torch.cat(senti_classifier_logits_list, dim=0)
            assert len(senti_classifier_logits) == len(starts)

            example["spans"] = []
            for _i in range(num_spans):
                distribution = senti_classifier_logits[_i]
                assert ends[_i] <= len(input_ids)
                example["spans"].append([int(starts[_i]), int(ends[_i]), distribution.detach().cpu().numpy().tolist()])
        with open(save_path + "/wordnet_bias_distillation_" + type + ".json", "w") as f:
            json.dump(absa_dataset, f)


# aspect-sentence concatenation for ABSA and sentence-level classification
def train(args, train_dataset, model, tokenizer, eval_dataset=None, eval_fn=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    # learning setup
    train_dataloader = setup_training_step(
        args, train_dataset, collate_fn=train_dataset.data_collate_fn)
    model, optimizer, scheduler = setup_opt(args, model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)

    global_step = 0
    best_accu = -1e5
    ma_dict = MovingAverageDict()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _idx_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps),
                              disable=args.local_rank not in [-1, 0])
        step_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # assert len(batch) == 5
            feed_dict = train_dataset.batch2feed_dict(batch)
            if args.model_class == "roberta":
                feed_dict.pop("token_type_ids")
            outputs = model(**feed_dict)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = update_wrt_loss(args, model, optimizer, loss)

            step_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                model_update_wrt_gradient(args, model, optimizer, scheduler)
                global_step += 1
                # update loss for logging
                ma_dict({"loss": step_loss})
                tb_writer.add_scalar("training_loss", step_loss, global_step)
                step_loss = 0.

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info(ma_dict.get_val_str())

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(args.output_dir, model, tokenizer, args)

                if eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    cur_accu = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step, tb_writer = tb_writer)
                    if cur_accu > best_accu:
                        best_accu = cur_accu
                        save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
        # evaluation each epoch or last epoch
        if (_idx_epoch == int(args.num_train_epochs) - 1) or (eval_dataset is not None and args.eval_steps <= 0):
            cur_accu = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step, tb_writer = tb_writer)
            if cur_accu > best_accu:
                best_accu = cur_accu
                save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
            fp.write("{}{}".format(best_accu, os.linesep))

def evaluate(
        args, eval_dataset, model, tokenizer, global_step,
        is_saving_pred=False, verbose=False, file_prefix="", tb_writer=None):
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader = setup_eval_step(
        args, eval_dataset, collate_fn=eval_dataset.data_collate_fn, )
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    _idx_ex = 0
    labels_list = []
    confs_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(_t.to(args.device) for _t in batch)
        feed_dict = eval_dataset.batch2feed_dict(batch)
        if args.model_class == "roberta":
            feed_dict.pop("token_type_ids")
        with torch.no_grad():
            outputs = model(**feed_dict)
            tmp_eval_loss, confs = outputs[:2]
            confs = torch.softmax(confs, dim=-1)
            # tmp_eval_loss, confs1, confs2 = outputs[:3]
            # confs1 = torch.softmax(confs1, dim=-1)
            # confs2 = torch.softmax(confs2, dim=-1)
            # confs = confs1.clone()
            # # ---- combination 1 -----
            # confs[:, 0] = confs1[:, 0] * confs2[:, 0]
            # confs[:, 1] = confs1[:, 1] * confs2[:, 1]
            # confs[:, 2] = confs1[:, 2] * confs2[:, 0]
            # ---- combination 2 -----
            # for _i, (p_n, neutral) in enumerate(confs2):
            #     if neutral > p_n:
            #         confs[_i][1] = 1

        eval_loss += tmp_eval_loss.mean().item()

        labels_list.append(feed_dict["labels"].detach().cpu().numpy())
        confs_list.append(confs.detach().cpu().numpy())

        nb_eval_examples += feed_dict["input_ids"].size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    labels = np.concatenate(labels_list, axis=0).astype("int64")
    confs = np.concatenate(confs_list, axis=0)

    preds = np.argmax(confs, axis=-1)   # (confs[:,1] > thresh).astype("int64")
    accu = np.mean(labels == preds)

    # accu = metrics.accuracy_score(labels, preds)
    import sklearn
    recall = sklearn.metrics.recall_score(labels, preds, average="macro")
    precision = sklearn.metrics.precision_score(labels, preds, average="macro")
    f1 = sklearn.metrics.f1_score(labels, preds, average="macro")

    result = {
        'eval_loss': eval_loss,
        "accuracy": accu,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }
    output_eval_file = os.path.join(args.output_dir, file_prefix + "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logging.info("***** Eval results at {}*****".format(global_step))
        writer.write("***** Eval results at {}*****\n".format(global_step))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, result[key], global_step)
        writer.write("\n")

    if is_saving_pred:
        sent_id_list = []
        sent_text_list = []
        term_list = []
        for absa_example in eval_dataset.example_list:
            sent_id_list.append(absa_example["sent_id"])
            sent_text_list.append(absa_example["sent_text"])
            term_list.append(absa_example["terms"][0]["term"] or "")

        assert len(sent_id_list) == labels.shape[0] == confs.shape[0]
        if args.nums_label == 3:
            pred_data = {
                "sent_id": sent_id_list,
                "sent_text": sent_text_list,
                "term": term_list,
                "label": labels,
                "negative": confs[:, 0],
                "neutral": confs[:, 1],
                "positive": confs[:, 2],
                "prediction": np.argmax(confs, axis=1),
            }
        elif args.nums_label == 2:
            pred_data = {
                "sent_id": sent_id_list,
                "sent_text": sent_text_list,
                "term": term_list,
                "label": labels,
                "pos&neg": confs[:, 0],
                "neutral": confs[:, 1],
                "prediction": np.argmax(confs, axis=1),
                "correct": labels == np.argmax(confs, axis=1)
            }
        pred_data_df = pd.DataFrame(pred_data)
        pred_data_df.to_csv(os.path.join(args.output_dir, file_prefix + "pred_data.csv"))

    return f1

# ---------------------------

def get_span(tokenizer, sent_text, wps_start, wps_end):
    if wps_start == -1:
        return "dummy span"
    sent_tokens = split_to_token(tokenizer, sent_text)
    charidx2tokenidx = parse_tk_idx_list_wrt_char(tokenizer, sent_text, sent_tokens)
    wp_list, id_list, wpidx2tokenidx = continue_tokenize_for_wordpiece(tokenizer, sent_tokens)
    char_start = charidx2tokenidx.index(wpidx2tokenidx[wps_start])
    # wps_end = wpidx2tokenidx[-1] if wps_end > wpidx2tokenidx[-1] else wps_end
    wps_end -= 1
    char_end = [idx for (idx, value) in enumerate(charidx2tokenidx) if value == wpidx2tokenidx[wps_end]][-1]
    return sent_text[char_start:char_end+1]

def evaluate_multi(
        args, eval_dataset, model, tokenizer, global_step,
        is_saving_pred=False, is_saving_disc_pred=True, verbose=False, file_prefix="", tb_writer=None, loss_figure=False):
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader = setup_eval_step(
        args, eval_dataset, collate_fn=eval_dataset.data_collate_fn_multi)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    cls_eval_loss, disc_eval_loss, adv_eval_loss, kl_eval_loss = 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    _idx_ex = 0
    labels_list = []
    confs_list = []
    bias_labels_list = []
    disc_confs_list = []
    cls_confs_list = []
    span_start_idx_list, span_end_idx_list, span_atten_weights_list, \
    span_total_scores_list, span_senti_scores_list, span_depend_scores_list = [], [], [], [], [], []

    # do evaluation
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(_t.to(args.device) for _t in batch)
        feed_dict = eval_dataset.batch2feed_dict_multi(batch, concat_way=args.concat_way)
        if args.model_class == "roberta":
            feed_dict.pop("token_type_ids")
        with torch.no_grad():
            if args.concat_way == 'adv':
                if args.save_spans_info:
                    aspects_emb, labels, pure_asp_embs, bias_labels, \
                    span_start_idx, span_end_idx, span_atten_weights,\
                    span_total_scores, span_senti_scores, span_depend_scores = model.get_forward_data(**feed_dict)
                else:
                    aspects_emb, labels, pure_asp_embs, bias_labels = model.get_forward_data(**feed_dict)
                tmp_eval_loss, confs, labels, tmp_cls_loss, tmp_adv_loss = model.encoder_forward(aspects_emb, labels, pure_asp_embs)
                tmp_disc_eval_loss, disc_confs = model.discriminator_forward(pure_asp_embs, bias_labels)[:2]
                disc_confs = torch.softmax(disc_confs, dim=-1)
                _, cls_confs, _ = model.encoder_forward(pure_asp_embs, labels, pure_asp_embs)[:3]
                cls_confs = torch.softmax(cls_confs, dim=-1)
            elif args.concat_way == 'opinion':
                if args.save_spans_info:
                    aspects_emb, labels, pure_asp_embs, bias_labels, \
                    span_start_idx, span_end_idx, span_atten_weights,\
                    span_total_scores, span_senti_scores, span_depend_scores, gold_loss = model.get_forward_data(**feed_dict)
                else:
                    aspects_emb, labels, pure_asp_embs, bias_labels = model.get_forward_data(**feed_dict)
                tmp_eval_loss, confs, labels, tmp_cls_loss, tmp_adv_loss, tmp_gold_loss = model.encoder_forward(aspects_emb, labels, pure_asp_embs, gold_loss)
                tmp_disc_eval_loss, disc_confs = model.discriminator_forward(pure_asp_embs, bias_labels)[:2]
                disc_confs = torch.softmax(disc_confs, dim=-1)
                _, cls_confs, _ = model.encoder_forward(pure_asp_embs, labels, pure_asp_embs, gold_loss)[:3]
                cls_confs = torch.softmax(cls_confs, dim=-1)
            elif args.concat_way == "adv_distillation" or args.concat_way == 'adv_supervision' or args.concat_way == "adv_supervision_treespan":
                if args.save_spans_info:
                    aspects_emb, labels, pure_asp_embs, bias_labels, kl_loss, \
                    span_start_idx, span_end_idx, span_atten_weights,\
                    span_total_scores, span_senti_scores, span_depend_scores = model.get_forward_data(**feed_dict)
                else:
                    aspects_emb, labels, pure_asp_embs, bias_labels = model.get_forward_data(**feed_dict)
                tmp_eval_loss, confs, labels, tmp_cls_loss, tmp_adv_loss, tmp_kl_loss = model.encoder_forward(aspects_emb, labels, pure_asp_embs, kl_loss, bias_labels)
                tmp_disc_eval_loss, disc_confs = model.discriminator_forward(pure_asp_embs, bias_labels)[:2]
                disc_confs = torch.softmax(disc_confs, dim=-1)
                _, cls_confs, _ = model.encoder_forward(pure_asp_embs, labels, pure_asp_embs, kl_loss, bias_labels)[:3]
                cls_confs = torch.softmax(cls_confs, dim=-1)
            else:
                outputs = model(**feed_dict)

            if args.save_spans_info:
                # tmp_eval_loss, confs, labels, span_start_idx, span_end_idx, span_atten_weights, span_total_scores, span_senti_scores, span_depend_scores = outputs
                if args.concat_way == "neu":
                    tmp_eval_loss, confs, labels, cls_loss, adv_loss, disc_loss, \
                    span_start_idx, span_end_idx, span_atten_weights, span_total_scores, span_senti_scores, span_depend_scores = outputs
                elif args.concat_way == "hoi":
                    tmp_eval_loss, confs, labels, cls_loss, pure_cls_loss, \
                    span_start_idx, span_end_idx, span_atten_weights, span_total_scores, span_senti_scores, span_depend_scores = outputs
            else:
                tmp_eval_loss, confs, labels = outputs[:3]
            confs = torch.softmax(confs, dim=-1)

        eval_loss += tmp_eval_loss.mean().item()
        if args.concat_way == "adv" or args.concat_way == 'opinion':
            cls_eval_loss += tmp_cls_loss.mean().item()
            disc_eval_loss += tmp_disc_eval_loss.mean().item()
            adv_eval_loss += tmp_adv_loss.mean().item()
        elif args.concat_way == "adv_distillation" or args.concat_way == 'adv_supervision' or args.concat_way == "adv_supervision_treespan":
            cls_eval_loss += tmp_cls_loss.mean().item()
            disc_eval_loss += tmp_disc_eval_loss.mean().item()
            adv_eval_loss += tmp_adv_loss.mean().item()
            kl_eval_loss += tmp_kl_loss.mean().item()

        labels_list.append(labels.detach().cpu().numpy())
        confs_list.append(confs.detach().cpu().numpy())

        if args.concat_way == 'adv' or args.concat_way == 'opinion' or args.concat_way == "adv_distillation" or args.concat_way == 'adv_supervision' or args.concat_way == "adv_supervision_treespan":
            bias_labels_list.append(bias_labels.detach().cpu().numpy())
            disc_confs_list.append(disc_confs.detach().cpu().numpy())
            cls_confs_list.append(cls_confs.detach().cpu().numpy())
        if args.save_spans_info:
            span_start_idx_list.append(span_start_idx.detach().cpu().numpy())
            span_end_idx_list.append(span_end_idx.detach().cpu().numpy())
            span_atten_weights_list.append(span_atten_weights.detach().cpu().numpy())
            span_total_scores_list.append(span_total_scores.detach().cpu().numpy())
            span_senti_scores_list.append(span_senti_scores.detach().cpu().numpy())
            span_depend_scores_list.append(span_depend_scores.detach().cpu().numpy())

        nb_eval_examples += feed_dict["input_ids"].size(0)
        nb_eval_steps += 1



    eval_loss = eval_loss / nb_eval_steps
    labels = np.concatenate(labels_list, axis=0).astype("int64")
    confs = np.concatenate(confs_list, axis=0)

    if args.save_spans_info:
        span_start_idx = np.concatenate(span_start_idx_list, axis=0)
        span_end_idx = np.concatenate(span_end_idx_list, axis=0)
        span_atten_weights = np.concatenate(span_atten_weights_list, axis=0)
        span_total_scores = np.concatenate(span_total_scores_list, axis=0)
        span_senti_scores = np.concatenate(span_senti_scores_list, axis=0)
        span_depend_scores = np.concatenate(span_depend_scores_list, axis=0)

    if args.concat_way == 'adv' or args.concat_way == 'opinion' or args.concat_way == "adv_distillation" or args.concat_way == 'adv_supervision' or args.concat_way == "adv_supervision_treespan":
        # record the eval loss
        cls_eval_loss = cls_eval_loss / nb_eval_steps
        disc_eval_loss = disc_eval_loss / nb_eval_steps
        adv_eval_loss = adv_eval_loss / nb_eval_steps
        if args.concat_way == "adv_distillation" or args.concat_way == 'adv_supervision' or args.concat_way == "adv_supervision_treespan":
            kl_eval_loss = kl_eval_loss / nb_eval_steps
        # record the discrimination prediction
        disc_confs = np.concatenate(disc_confs_list, axis=0)
        disc_preds = np.argmax(disc_confs, axis=-1)  # (confs[:,1] > thresh).astype("int64")
        bias_labels = np.concatenate(bias_labels_list, axis=0).astype("int64")
        cls_confs = np.concatenate(cls_confs_list, axis=0)
        cls_preds = np.argmax(cls_confs, axis=-1)

    preds = np.argmax(confs, axis=-1)   # (confs[:,1] > thresh).astype("int64")
    accu = np.mean(labels == preds)

    # accu = metrics.accuracy_score(labels, preds)
    import sklearn
    recall = sklearn.metrics.recall_score(labels, preds, average="macro")
    precision = sklearn.metrics.precision_score(labels, preds, average="macro")
    f1 = sklearn.metrics.f1_score(labels, preds, average="macro")

    result = {
        'eval_loss': eval_loss,
        "accuracy": accu,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }

    if args.concat_way == 'adv' or args.concat_way == 'opinion' or args.concat_way == "adv_distillation" or args.concat_way == 'adv_supervision' or args.concat_way == "adv_supervision_treespan":
        result["cls_eval_loss"] = cls_eval_loss
        result["disc_eval_loss"] = disc_eval_loss
        result["adv_eval_loss"] = adv_eval_loss
        if args.concat_way == "adv_distillation" or args.concat_way == 'adv_supervision' or args.concat_way == "adv_supervision_treespan":
            result["kl_eval_loss"] = kl_eval_loss

    if is_saving_pred:
        sent_id_list = []
        sent_text_list = []
        term_list = []
        for absa_example in eval_dataset.example_list:
            for term_dict in absa_example["terms"]:
                sent_id_list.append(absa_example["sent_id"])
                sent_text_list.append(absa_example["sent_text"])
                term_list.append(term_dict["term"])

        assert len(sent_id_list) == labels.shape[0] == confs.shape[0]
        pred_data = {
            "sent_id": sent_id_list,
            "sent_text": sent_text_list,
            "term": term_list,
            "label": labels,
            "negative": confs[:, 0],
            "neutral": confs[:, 1],
            "positive": confs[:, 2],
            "prediction":np.argmax(confs,axis=1),
        }
        pred_data_df = pd.DataFrame(pred_data)
        pred_data_df.to_csv(os.path.join(args.output_dir, file_prefix + "pred_data.csv"))

    if is_saving_disc_pred:
        sent_id_list = []
        sent_text_list = []
        term_list = []
        for absa_example in eval_dataset.example_list:
            for term_dict in absa_example["terms"]:
                sent_id_list.append(absa_example["sent_id"])
                sent_text_list.append(absa_example["sent_text"])
                term_list.append(term_dict["term"])

        assert len(sent_id_list) == labels.shape[0] == confs.shape[0]
        pred_data = {
            "sent_id": sent_id_list,
            "sent_text": sent_text_list,
            "term": term_list,
            "bias_label": bias_labels,
            "disc_negative": disc_confs[:, 0],
            "disc_neutral": disc_confs[:, 1],
            "disc_positive": disc_confs[:, 2],
            "disc_prediction":disc_preds,
            "disc_correct": bias_labels == disc_preds,
            "labels": labels,
            "cls_negative": cls_confs[:, 0],
            "cls_neutral": cls_confs[:, 1],
            "cls_positive": cls_confs[:, 2],
            "cls_prediction":cls_preds,
            "cls_correct":cls_preds == labels
        }
        pred_data_df = pd.DataFrame(pred_data)
        pred_data_df.to_csv(os.path.join(args.output_dir, file_prefix + "disc_pred_data.csv"))

        if file_prefix == 'final_test_':
            no_span_accu = np.mean(labels == cls_preds)
            no_span_recall = sklearn.metrics.recall_score(labels, cls_preds, average="macro")
            no_span_precision = sklearn.metrics.precision_score(labels, cls_preds, average="macro")
            no_span_f1 = sklearn.metrics.f1_score(labels, cls_preds, average="macro")

            no_span_output_eval_file = os.path.join(args.output_dir, file_prefix + "no_span_results.txt")
            with open(no_span_output_eval_file, "a") as writer:
                logging.info("***** Test results at {}*****".format(global_step))
                writer.write("%s = %s\n" % ("no_span_accu", str(no_span_accu)))
                writer.write("%s = %s\n" % ("no_span_precision", str(no_span_precision)))
                writer.write("%s = %s\n" % ("no_span_recall", str(no_span_recall)))
                writer.write("%s = %s\n" % ("no_span_f1", str(no_span_f1)))
                writer.write("\n")

    if args.save_spans_info:
        sent_id_list = []
        sent_text_list = []
        term_list = []
        depend_span_list = []
        case_idx = 0
        for _idx, absa_example in enumerate(eval_dataset.example_list):
            for term_dict in absa_example["terms"]:
                sent_id_list.append(absa_example["sent_id"])
                sent_text_list.append(absa_example["sent_text"])
                term_list.append(term_dict["term"])
                for _i in range(args.top_n):
                    depend_span_list.append(get_span(tokenizer, absa_example["sent_text"], span_start_idx[case_idx][_i], span_end_idx[case_idx][_i]))
                case_idx += 1

        assert len(sent_id_list) == labels.shape[0] == confs.shape[0]
        prediction = np.argmax(confs,axis=1)
        pred_data = {
            "sent_id": sent_id_list,
            "sent_text": sent_text_list,
            "term": term_list,
            "label": labels,
            "negative": confs[:, 0],
            "neutral": confs[:, 1],
            "positive": confs[:, 2],
            "prediction":prediction,
            "correct":labels==prediction,
        }

        for _i in range(args.top_n):
            pred_data["depend_span_" + str(_i)] = [depend_span_list[_i+_j] for _j in range(0, len(depend_span_list), args.top_n)]
            pred_data["atten_weight_" + str(_i)] = span_atten_weights[:, _i]
            pred_data["total_score_" + str(_i)] = span_total_scores[:, _i]
            pred_data["senti_score_" + str(_i)] = span_senti_scores[:, _i]
            pred_data["dependency_score_" + str(_i)] = span_depend_scores[:, _i]

        pred_data_df = pd.DataFrame(pred_data)
        pred_data_df.to_csv(os.path.join(args.output_dir, file_prefix + "interpretability.csv"))

    output_eval_file = os.path.join(args.output_dir, file_prefix + "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logging.info("***** Eval results at {}*****".format(global_step))
        writer.write("***** Eval results at {}*****\n".format(global_step))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, result[key], global_step)
        writer.write("\n")

    if args.eval_interpretable and file_prefix == 'final_test_':
        _i = 0
        for absa_example in eval_dataset.example_list:
            for term_dict in absa_example["terms"]:
                term_dict["ranked_spans"] = [span.strip() for span in depend_span_list[_i:_i + args.top_n]]
                _i += args.top_n

        for _absa_idx, _absa_sample in enumerate(eval_dataset.absa_dataset):
            for _example in eval_dataset.example_list:
                if _absa_sample["sent_id"] == _example["sent_id"]:
                    eval_dataset.absa_dataset[_absa_idx] = _example
        if "twitter" not in args.dataset:
            char_evaluate_interpretable(args, eval_dataset.absa_dataset, output_eval_file)
    return f1

def getNumofCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0
    p = 0

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return str1[p - maxNum:p], maxNum

def char_evaluate_interpretable(args, eval_dataset, output_eval_file):
    dataset_name = args.dataset.split("_")[0] + "_wordnet_bias"
    _, _, opinion_set = get_raw_datasets(dataset_name + "_opinions")

    for topn in [1, 3, 5, 10]:
        num_match = 0
        num_asps = 0
        num_opinion_spans = 0
        no_match = 0
        precision_list = []
        recall_list = []
        num_asps_neutral_noOT = 0
        num_asps_neutral_noOT_dummy = 0
        for sent_idx in range(len(eval_dataset)):

            spans_exp = eval_dataset[sent_idx]["terms"]

            opinion_exp = opinion_set[sent_idx]
            assert eval_dataset[sent_idx]["sent_id"] == opinion_exp["sent_id"]

            if opinion_exp["terms"] == []:
                pass
            else:
                for _term_idx, term_dict in enumerate(opinion_exp["terms"]):

                    tmp_pre = []
                    tmp_recall = []
                    if "opinions" not in list(term_dict.keys()):
                        if term_dict["polarity"] == "neutral":
                            num_asps_neutral_noOT += 1
                            candi_spans_list = spans_exp[_term_idx]["ranked_spans"][:topn]
                            if "dummy span" in candi_spans_list:
                                num_asps_neutral_noOT_dummy += 1
                    else:
                        num_asps += 1
                        candi_spans_list = spans_exp[_term_idx]["ranked_spans"][:topn]
                        for opinion in term_dict["opinions"]:
                            num_opinion_spans += 1
                            if opinion["op_term"] in candi_spans_list:  # exactly match
                                num_match += 1
                                tmp_pre.append(1)
                                tmp_recall.append(1)
                            else:
                                _best_pre, _best_recall = 0, 0
                                for _span_idx, span in enumerate(candi_spans_list):
                                    max_overlap = getNumofCommonSubstr(span, opinion["op_term"])[1]
                                    _tmp_pre = 0 if len(span) == 0 else max_overlap/len(span)
                                    _tmp_rec = 0 if len(opinion["op_term"]) == 0 else max_overlap/len(opinion["op_term"])
                                    if _best_pre < _tmp_pre:
                                        _best_pre = _tmp_pre
                                        _best_recall = _tmp_rec

                                if _best_pre == 0:
                                    no_match += 1
                                tmp_pre.append(_best_pre)
                                tmp_recall.append(_best_recall)

                        num_opinions = len(tmp_pre)
                        final_pre, final_recall = 0, 0
                        for _idx in range(num_opinions):
                            final_pre += tmp_pre[_idx]
                            final_recall += tmp_recall[_idx]
                        final_pre = final_pre / num_opinions
                        final_recall = final_recall / num_opinions
                        precision_list.append(final_pre)
                        recall_list.append(final_recall)


        precision_list = np.array(precision_list)
        recall_list = np.array(recall_list)
        assert len(precision_list) == num_asps
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1_supplication = np.array([0] * np.sum(recall_list == 0))
        f1 = np.mean(np.concatenate((2 * precision_list[recall_list != 0] * recall_list[recall_list != 0] / (
                        precision_list[recall_list != 0] + recall_list[recall_list != 0]),  f1_supplication), axis=0))
        with open(output_eval_file, "a") as writer:

            writer.write("--------- Top-" + str(topn) + " results ---------\n")
            writer.write("total match: " + str(num_match)+ "\n")
            writer.write("em match ratio: " + str(num_match / num_opinion_spans)+ "\n")
            writer.write("no match: " + str(no_match)+ "\n")
            writer.write("precision: " + str(precision)+ "\n")
            writer.write("recall: " + str(recall)+ "\n")
            writer.write("f1: " + str(f1)+ "\n")
            writer.write("num_asps_neutral_noOT: " + str(num_asps_neutral_noOT) + "\n")
            writer.write("num_asps_neutral_noOT_dummy: " + str(num_asps_neutral_noOT_dummy) + "\n")
            writer.write("dummy_hits: " + str(num_asps_neutral_noOT_dummy / num_asps_neutral_noOT) + "\n\n\n")

        print("------- Top" + str(topn) + "results ---------")
        print("total num of aspects with explict opinion words: " + str(num_asps))
        print("total opinion words (a aspect term may have several opinion words): " + str(num_opinion_spans))
        print("total match: " + str(num_match))
        print("em match ratio: " + str(num_match / num_opinion_spans))
        print("no match: " + str(no_match))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("f1: " + str(f1))
        print("num_asps_neutral_noOT: " + str(num_asps_neutral_noOT))
        print("num_asps_neutral_noOT_dummy: " + str(num_asps_neutral_noOT_dummy))
        print("dummy_hits: " + str(num_asps_neutral_noOT_dummy / num_asps_neutral_noOT) + "\n")

# mlp
def train_multi(args, train_dataset, model, tokenizer, eval_dataset=None, eval_fn=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    # learning setup
    train_dataloader = setup_training_step(
        args, train_dataset, collate_fn=train_dataset.data_collate_fn_multi)
    model, optimizer, scheduler = setup_opt(args, model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)

    if args.concat_way == 'adv' or args.concat_way == 'opinion':
        is_saving_disc_pred = True
    else:
        is_saving_disc_pred = False

    global_step = 0
    best_accu = -1e5
    ma_dict = MovingAverageDict()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _idx_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps),
                              disable=args.local_rank not in [-1, 0])
        step_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # assert len(batch) == 5
            feed_dict = train_dataset.batch2feed_dict_multi(batch, concat_way=args.concat_way)
            if args.model_class == "roberta":
                feed_dict.pop("token_type_ids")
            outputs = model(**feed_dict)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = update_wrt_loss(args, model, optimizer, loss)

            step_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                model_update_wrt_gradient(args, model, optimizer, scheduler)
                global_step += 1
                # update loss for logging
                ma_dict({"loss": step_loss})
                if len(outputs) == 6:
                    tb_writer.add_scalar("cls_loss", outputs[3].item(), global_step)
                    tb_writer.add_scalar("hinge_loss", outputs[4].item(), global_step)
                    tb_writer.add_scalar("span_cls_loss", outputs[5].item(), global_step)
                if len(outputs) == 5 or len(outputs) == 11:
                    tb_writer.add_scalar("cls_loss", outputs[3].item(), global_step)
                    tb_writer.add_scalar("pure_cls_loss", outputs[4].item(), global_step)
                tb_writer.add_scalar("training_loss", step_loss, global_step)
                step_loss = 0.

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info(ma_dict.get_val_str())

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(args.output_dir, model, tokenizer, args)

                if eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    cur_accu = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step, tb_writer=tb_writer, is_saving_disc_pred=is_saving_disc_pred)
                    if cur_accu > best_accu:
                        best_accu = cur_accu
                        save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
        # evaluation each epoch or last epoch
        if (_idx_epoch == int(args.num_train_epochs) - 1) or (eval_dataset is not None and args.eval_steps <= 0):
            cur_accu = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step, tb_writer=tb_writer, is_saving_disc_pred=is_saving_disc_pred)
            if cur_accu > best_accu:
                best_accu = cur_accu
                save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
            fp.write("{}{}".format(best_accu, os.linesep))

# SARL
def train_adv(args, train_dataset, model, tokenizer, eval_dataset=None, eval_fn=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    # learning setup
    train_dataloader = setup_training_step(
        args, train_dataset, collate_fn=train_dataset.data_collate_fn_multi)
    model, encoder_optimizer, task_optimizer, disc_optimizer, encoder_scheduler, task_scheduler, disc_scheduler = adv_setup_opt(args, model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)

    global_step = 0
    best_accu = -1e5
    ma_dict = MovingAverageDict()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _idx_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps),
                              disable=args.local_rank not in [-1, 0])
        step_cls_and_adv_loss, step_disc_loss = 0, 0
        for step, batch in enumerate(epoch_iterator):
            if args.disc_ratio_decrease:
                if global_step > 1/3:
                    if global_step % 10 == 0:
                        args.discriminator_ratio -= 0.02

            # if step < 280:
            #     continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # assert len(batch) == 5
            feed_dict = train_dataset.batch2feed_dict_multi(batch, concat_way=args.concat_way)
            if args.model_class == "roberta":
                feed_dict.pop("token_type_ids")


            # get tensors used in forward
            if args.concat_way == "adv_distillation" or args.concat_way == "adv_supervision" or args.concat_way == "adv_supervision_treespan":
                aspects_emb, labels, pure_asp_embs, bias_labels, kl_loss = model.get_forward_data(**feed_dict)[:5]
            else:
                aspects_emb, labels, pure_asp_embs, bias_labels = model.get_forward_data(**feed_dict)[:4]

            # # ---------- View the gradients ------------------
            # para_dict_init = {}
            # for name, parms in model.named_parameters():
            #     if name == 'roberta.embeddings.word_embeddings.weight':
            #         para_dict_init[name] = parms.grad
            #     if name.split('.')[0] != 'roberta':
            #         para_dict_init[name] = parms.grad
            # # -----------------------------------


            # optimize the discriminator
            if not args.no_adversarial:
                if random() < args.discriminator_ratio:
                    disc_optimizer.zero_grad()
                    disc_loss = model.discriminator_forward(pure_asp_embs, bias_labels)[0]
                    disc_loss = disc_loss.mean()
                    step_disc_loss += disc_loss.item()
                    disc_loss.backward()
                    disc_optimizer.step()
                    disc_optimizer.zero_grad()
                    disc_scheduler.step()
                    ma_dict({"disc_loss": step_disc_loss})
                    tb_writer.add_scalar("disc_loss", step_disc_loss, global_step)

            # optimize the encoder
            if random() < args.encoder_ratio:
                encoder_optimizer.zero_grad()
                task_optimizer.zero_grad()
                if args.concat_way == "adv_distillation" or args.concat_way == "adv_supervision" or args.concat_way == "adv_supervision_treespan":
                    cls_and_adv_loss, logits, _, cls_loss, adv_loss, kl_loss = model.encoder_forward(aspects_emb, labels, pure_asp_embs, kl_loss, bias_labels)
                else:
                    cls_and_adv_loss, logits, _, cls_loss, adv_loss = model.encoder_forward(aspects_emb, labels, pure_asp_embs)
                cls_and_adv_loss = cls_and_adv_loss.mean()
                step_cls_and_adv_loss += cls_and_adv_loss.item()
                cls_and_adv_loss.backward()
                encoder_optimizer.step()
                task_optimizer.step()
                encoder_scheduler.step()
                task_scheduler.step()

                ma_dict({"cls_and_adv_loss": step_cls_and_adv_loss})
                ma_dict({"cls_loss": cls_loss.item()})
                ma_dict({"adv_loss": adv_loss.item()})
                if args.concat_way == "adv_distillation" or args.concat_way == "adv_supervision" or args.concat_way == "adv_supervision_treespan":
                    ma_dict({"kl_loss": kl_loss.item()})
                tb_writer.add_scalar("cls_and_adv_loss", step_cls_and_adv_loss, global_step)
                tb_writer.add_scalar("cls_loss", cls_loss.item(), global_step)
                tb_writer.add_scalar("adv_loss", adv_loss.item(), global_step)
                if args.concat_way == "adv_distillation" or args.concat_way == "adv_supervision"or args.concat_way == "adv_supervision_treespan":
                    tb_writer.add_scalar("kl_loss", kl_loss.item(), global_step)

            model.zero_grad()

            global_step += 1
            # update loss for logging
            # ma_dict({"cls_and_adv_loss": step_cls_and_adv_loss})
            # ma_dict({"disc_loss": step_disc_loss})
            # tb_writer.add_scalar("cls_and_adv_loss", step_cls_and_adv_loss, global_step)
            # tb_writer.add_scalar("disc_loss", step_disc_loss, global_step)
            # tb_writer.add_scalar("cls_loss", cls_loss.item(), global_step)
            # tb_writer.add_scalar("adv_loss", adv_loss.item(), global_step)
            step_cls_and_adv_loss, step_disc_loss = 0., 0.

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging.info(ma_dict.get_val_str())
                with open(os.path.join(args.output_dir, 'loss_record.txt'), 'a') as wf:
                    wf.write(str(global_step) + ': ')
                    wf.write(ma_dict.get_val_str())
                    wf.write('\n')

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model_with_default_name(args.output_dir, model, tokenizer, args)

            if eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                cur_accu = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step, tb_writer=tb_writer)
                if cur_accu > best_accu:
                    best_accu = cur_accu
                    save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

            if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
        # evaluation each epoch or last epoch
        if (_idx_epoch == int(args.num_train_epochs) - 1) or (eval_dataset is not None and args.eval_steps <= 0):
            cur_accu = eval_fn(args, eval_dataset, model, tokenizer, global_step=global_step, tb_writer=tb_writer)
            if cur_accu > best_accu:
                best_accu = cur_accu
                save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        with open(os.path.join(args.output_dir, "best_eval_results.txt"), "a") as fp:
            fp.write("training_epoch_" + str(_idx_epoch) + ": ")
            fp.write("{}{}".format(best_accu, os.linesep))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="roberta", type=str, help="[roberta|bert]")
    parser.add_argument("--dataset", default="rest14", type=str, help="[rest14|lap14|rest15|rest16|twitter]")
    parser.add_argument("--dataset_dir", default=None, type=str, help="")
    parser.add_argument("--concat_way", default="naive", type=str, help="")
    parser.add_argument("--data_format", default="term", type=str, help="")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--nums_label", default=3, type=int,
                        help="The number of labels.")

    parser.add_argument("--train_rate", default=1.0, type=float)
    parser.add_argument("--max_span_width", default=15, type=int)
    parser.add_argument("--max_rela_distance", default=100, type=int)
    parser.add_argument("--feature_emb_size", default=30, type=int)  # 32 if use multi-heads
    parser.add_argument("--use_width_features", action="store_true")
    parser.add_argument("--use_rela_dis_features", action="store_true")


    parser.add_argument("--top_n", default=10, type=int, help="top_n spans according to the total score")
    parser.add_argument("--use_dep_dis_features", action="store_true", help="use the feature about dependence distance between aspect term and other spans")

    parser.add_argument("--use_gate", action="store_true")
    parser.add_argument("--save_spans_info", action="store_true")

    # for discriminator/adversarial
    parser.add_argument("--encoder_lr", default=1e-5, type=float)
    parser.add_argument("--others_lr", default=1e-5, type=float)
    parser.add_argument("--phrase_sentiment_model_path", default=None, type=str,
                        help="Path to trained phrase sentiment model")
    parser.add_argument("--nums_bias_label", default=3, type=int,
                        help="The number of bias labels.")

    parser.add_argument("--adv_loss_weight", default=0.1, type=float)
    parser.add_argument("--discriminator_ratio", default=0.2, type=float)
    parser.add_argument("--encoder_ratio", default=1.0 , type=float)

    parser.add_argument("--eval_interpretable", action="store_true")
    parser.add_argument("--get_wordnet_bias", action="store_true")
    parser.add_argument("--get_distillation_dataset", action="store_true")

    parser.add_argument("--no_distillation", action="store_true")
    parser.add_argument("--no_sentiment_score", action="store_true")
    parser.add_argument("--no_adversarial", action="store_true")
    parser.add_argument("--no_adv_loss", action="store_true")
    parser.add_argument("--adversarial_on_polarity", action="store_true")
    parser.add_argument("--disc_ratio_decrease", action="store_true")



    define_hparams_training(parser)
    args = parser.parse_args()
    setup_prerequisite(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.data_format == "multi_aspects":
        # if args.concat_way == "hoi":
        #     if args.model_class == "roberta":
        #         model_class = RoBERTaHoiSpanCLS
        #         config_class = RobertaConfig
        #         tokenizer_class = RobertaTokenizer
        # elif args.concat_way == "adv":
        #     if args.model_class == "roberta":
        #         model_class = RoBERTaHoiSpanCLSAdv
        #         config_class = RobertaConfig
        #         tokenizer_class = RobertaTokenizer
        # elif args.concat_way == "neu":
        #     if args.model_class == "roberta":
        #         model_class = RoBERTaHoiSpanCLSNeu
        #         config_class = RobertaConfig
        #         tokenizer_class = RobertaTokenizer
        # elif args.concat_way == "opinion":
        #     if args.model_class == "roberta":
        #         model_class = RoBERTaHoiSpanCLSAdvOpinion
        #         config_class = RobertaConfig
        #         tokenizer_class = RobertaTokenizer
        # elif args.concat_way == "adv_supervision" or args.concat_way == "adv_supervision_treespan":
        #     if args.model_class == "roberta":
        #         model_class = RoBERTaHoiSpanCLSAdvSupervision
        #         config_class = RobertaConfig
        #         tokenizer_class = RobertaTokenizer
        #     elif args.model_class == "bert":
        #         model_class = BERTHoiSpanCLSAdvSupervision
        #         config_class = BertConfig
        #         tokenizer_class = BertTokenizer
        if args.concat_way == "adv_distillation":
            if args.model_class == "roberta":
                model_class = RoBERTaSpanCLSAdvDistillation
                config_class = RobertaConfig
                tokenizer_class = RobertaTokenizer
            elif args.model_class == "bert":
                model_class = BERTSpanCLSAdvDistillation
                config_class = BertConfig
                tokenizer_class = BertTokenizer
        else:
            if args.model_class == "bert":
                model_class = BertMultiAspectsCLS
                config_class = BertConfig
                tokenizer_class = BertTokenizer
            elif args.model_class == "roberta":
                model_class = RoBERTaMultiAspectsCLS
                config_class = RobertaConfig
                tokenizer_class = RobertaTokenizer
    elif args.data_format == "term_span":
        if args.model_class == "bert":
            raise NotImplementedError
        elif args.model_class == "roberta":
            model_class = RoBERTaSpanCLS
            config_class = RobertaConfig
            tokenizer_class = RobertaTokenizer
    else:
        if args.model_class == "bert":
            model_class = BertForSequenceClassification
            config_class = BertConfig
            tokenizer_class = BertTokenizer
        elif args.model_class == "roberta":
            model_class = RobertaForSequenceClassification
            config_class = RobertaConfig
            tokenizer_class = RobertaTokenizer
        else:
            raise NotImplementedError


    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.nums_label

    config.max_span_width = args.max_span_width
    config.max_rela_distance = args.max_rela_distance
    config.feature_emb_size = args.feature_emb_size
    config.use_width_features = args.use_width_features
    config.use_rela_dis_features = args.use_rela_dis_features
    config.top_n = args.top_n
    config.use_dep_dis_features = args.use_dep_dis_features
    config.use_gate = args.use_gate
    config.save_spans_info = args.save_spans_info
    config.phrase_sentiment_model_path = args.phrase_sentiment_model_path

    config.nums_bias_label = args.nums_bias_label
    config.adv_loss_weight = args.adv_loss_weight

    config.no_distillation = args.no_distillation
    config.no_sentiment_score = args.no_sentiment_score
    config.no_adversarial = args.no_adversarial
    config.no_adv_loss = args.no_adv_loss
    config.adversarial_on_polarity = args.adversarial_on_polarity



    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    if args.get_distillation_dataset:
        get_distillation_dataset(args, config, tokenizer, args.dataset)
        return

    logger.info("Training/evaluation parameters %s", args)

    # get wordnet sentiment bias datasets
    if args.get_wordnet_bias:
        for dataset in ['lap14', 'rest14', 'rest15', 'rest16', 'twitter']:
            train_dataset_raw, dev_dataset_raw, test_dataset_raw = get_raw_datasets(dataset)
            dataset_list = [train_dataset_raw, test_dataset_raw]
            datatype_list = ['train', 'test']
            for _i in range(len(dataset_list)):
                get_wordnet_bias(args, dataset_list[_i], dataset, datatype_list[_i])
        logger.info("Get wordnet sentiment bias finished!")
        return

    train_dataset_raw, dev_dataset_raw, test_dataset_raw = get_raw_datasets(
        args.dataset, args.dataset_dir)
    train_dataset = AbsaDataset(config, train_dataset_raw, args.data_format, "train", args.model_class, tokenizer,
                                args.max_seq_length,
                                concat_way=args.concat_way, nums_label=args.nums_label, rate=args.train_rate,
                                nums_bias_label=args.nums_bias_label,
                                max_span_width=args.max_span_width)
    dev_dataset = AbsaDataset(config, dev_dataset_raw, args.data_format, "dev", args.model_class, tokenizer,
                              args.max_seq_length,
                              concat_way=args.concat_way, nums_label=args.nums_label,
                              nums_bias_label=args.nums_bias_label,
                              max_span_width=args.max_span_width)
    test_dataset = AbsaDataset(config, test_dataset_raw, args.data_format, "test", args.model_class, tokenizer,
                               args.max_seq_length,
                               concat_way=args.concat_way, nums_label=args.nums_label,
                               nums_bias_label=args.nums_bias_label,
                               max_span_width=args.max_span_width)

    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    # if args.concat_way == "sst_pre":
    #     if args.do_train or args.do_eval:
    #         raise ValueError
    #     if args.do_prediction:
    #         sst_evaluate(args, train_dataset, model, tokenizer, global_step=None, verbose=True,
    #                      file_prefix="final_trainset_test_", is_saving_pred=True, dataset_type="train")
    #         sst_evaluate(args, dev_dataset, model, tokenizer, global_step=None, verbose=True,
    #                      file_prefix="final_devset_test_", is_saving_pred=True, dataset_type="dev")
    #         sst_evaluate(args, test_dataset, model, tokenizer, global_step=None, verbose=True,
    #                      file_prefix="final_testset_test_", is_saving_pred=True, dataset_type="test")
    # elif args.concat_way == "term_only":
    #     if args.do_train or args.do_eval:
    #         raise ValueError
    #     if args.do_prediction:
    #         get_bias_evaluate(args, train_dataset, model, tokenizer, global_step=None, verbose=True,
    #                      file_prefix="final_trainset_test_", is_saving_pred=True, dataset=args.dataset, dataset_type="train")
    #         get_bias_evaluate(args, dev_dataset, model, tokenizer, global_step=None, verbose=True,
    #                      file_prefix="final_devset_test_", is_saving_pred=True, dataset=args.dataset,dataset_type="dev")
    #         get_bias_evaluate(args, test_dataset, model, tokenizer, global_step=None, verbose=True,
    #                      file_prefix="final_testset_test_", is_saving_pred=True, dataset=args.dataset,dataset_type="test")

    # elif args.data_format != "multi_aspects" and args.concat_way != "naive_multi" and args.concat_way != "naive_twitter_multi" and args.concat_way != "naive_sampling_spans":
    if args.data_format != "multi_aspects":
        if args.do_train:
            train(args, train_dataset, model, tokenizer, dev_dataset, eval_fn=evaluate)
        if args.do_eval or args.do_prediction:
            if args.do_train:
                model = model_class.from_pretrained(args.output_dir, config=config)
                model.to(args.device)
            if args.fp16:
                model = setup_eval_model_for_fp16(args, model)

            evaluate(args, dev_dataset, model, tokenizer, global_step=None, verbose=True, file_prefix="final_dev_",
                     is_saving_pred=True)
            evaluate(args, test_dataset, model, tokenizer, global_step=None, verbose=True,
                         file_prefix="final_test_", is_saving_pred=True)
    # elif args.concat_way == "adv" or args.concat_way == "adv_supervision" or args.concat_way == "adv_supervision_treespan":
    #     if args.do_train:
    #         train_adv(args, train_dataset, model, tokenizer, dev_dataset, eval_fn=evaluate_multi)
    #     if args.do_eval or args.do_prediction:
    #         if args.do_train:
    #             model = model_class.from_pretrained(args.output_dir, config=config)
    #             model.to(args.device)
    #         if args.fp16:
    #             model = setup_eval_model_for_fp16(args, model)
    #         evaluate_multi(args, test_dataset, model, tokenizer, global_step=None, verbose=True, file_prefix="final_test_",
    #                  is_saving_pred=True)
    elif args.concat_way == "adv_distillation":
        if args.do_train:
            train_adv(args, train_dataset, model, tokenizer, dev_dataset, eval_fn=evaluate_multi)
        if args.do_eval or args.do_prediction:
            if args.do_train:
                model = model_class.from_pretrained(args.output_dir, config=config)
                model.to(args.device)
            if args.fp16:
                model = setup_eval_model_for_fp16(args, model)
            evaluate_multi(args, test_dataset, model, tokenizer, global_step=None, verbose=True, file_prefix="final_test_",
                     is_saving_pred=True)
    # elif args.concat_way == "opinion":
    #     if args.do_train:
    #         train_opinion(args, train_dataset, model, tokenizer, dev_dataset, eval_fn=evaluate_multi)
    #     if args.do_eval or args.do_prediction:
    #         if args.do_train:
    #             model = model_class.from_pretrained(args.output_dir, config=config)
    #             model.to(args.device)
    #         if args.fp16:
    #             model = setup_eval_model_for_fp16(args, model)
    #         evaluate_multi(args, test_dataset, model, tokenizer, global_step=None, verbose=True, file_prefix="final_test_",
    #                  is_saving_pred=True)
    # elif args.concat_way == "neu":
    #     if args.do_train:
    #         train_adv_old(args, train_dataset, model, tokenizer, dev_dataset, eval_fn=evaluate_multi)
    #     if args.do_eval or args.do_prediction:
    #         if args.do_train:
    #             model = model_class.from_pretrained(args.output_dir, config=config)
    #             model.to(args.device)
    #         if args.fp16:
    #             model = setup_eval_model_for_fp16(args, model)
    #         evaluate_multi(args, test_dataset, model, tokenizer, global_step=None, verbose=True, file_prefix="final_test_",
    #                  is_saving_pred=True)
    else:
        if args.do_train:
            train_multi(args, train_dataset, model, tokenizer, dev_dataset, eval_fn=evaluate_multi)
        if args.do_eval or args.do_prediction:
            if args.do_train:
                model = model_class.from_pretrained(args.output_dir, config=config)
                model.to(args.device)
            if args.fp16:
                model = setup_eval_model_for_fp16(args, model)

            evaluate_multi(args, test_dataset, model, tokenizer, global_step=None, verbose=True, file_prefix="final_test_",
                     is_saving_pred=True, is_saving_disc_pred=False)



if __name__ == '__main__':
    main()
