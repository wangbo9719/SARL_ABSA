from torch.utils.data import Dataset
from lxml import etree
from peach.utils_tokenizer import split_to_token, parse_tk_idx_list_wrt_char, continue_tokenize_for_wordpiece, char_to_token_span
import torch, os
import json
from nltk.stem import WordNetLemmatizer
import random
import spacy
nlp = spacy.load("en_core_web_sm")
import networkx as nx

wnl = WordNetLemmatizer()

def load_list_from_file(file_path):
    data = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                data.append(line.strip())
    return data

def read_semeval14_absa_xml(xml_path):
    absa_dataset = []

    with open(xml_path, 'rb') as f:
        raw = f.read()
    root = etree.fromstring(raw)
    for sentence in root:
        sample = {}
        # get sent id
        sample["sent_id"] = sentence.attrib["id"]
        sample["sent_text"] = sentence.find('text').text
        # items
        sample["terms"] = []
        terms = sentence.find('aspectTerms')
        if terms is not None:
            for term in terms:
                term_dict = dict(term.attrib)
                term_dict["from"] = int(term_dict["from"])
                term_dict["to"] = int(term_dict["to"])
                sample["terms"].append(term_dict)
        # categories
        sample["categories"] = []
        categories = sentence.find('aspectCategories')
        if categories is not None:
            for category in categories:
                sample["categories"].append(dict(category.attrib))

        absa_dataset.append(sample)
    return absa_dataset

def read_bias_absa_json(json_path):
    with open(json_path, 'r') as f:
        absa_dataset = json.load(f)
    return absa_dataset

def read_rest15_16_absa_xml(xml_path):
    absa_dataset = []

    with open(xml_path, 'rb') as f:
        raw = f.read()
    root = etree.fromstring(raw)
    for review in root:
        for sentences in review:
            for sentence in sentences:
                sample = {}
                # get sent id
                sample["sent_id"] = sentence.attrib["id"]
                sample["sent_text"] = sentence.find('text').text
                # items
                sample["terms"] = []
                terms = sentence.find('Opinions')
                if terms is not None:
                    terms_set = set()
                    for term in terms:
                        term_dict = dict(term.attrib)
                        if term_dict["target"] not in terms_set:
                            term_dict["term"] = term_dict.pop("target")
                            term_dict["from"] = int(term_dict["from"])
                            term_dict["to"] = int(term_dict["to"])
                            if term_dict["term"] != 'NULL':
                                sample["terms"].append(term_dict)
                                terms_set.add(term_dict["term"])

                absa_dataset.append(sample)
    return absa_dataset

def read_twitter_absa_dataset(data_path):
    absa_dataset = []

    raw_data = load_list_from_file(data_path)
    assert len(raw_data) % 3 == 0

    for i in range(0, len(raw_data), 3):
        sent_text_with_ph = raw_data[i]
        term = raw_data[i+1]
        polarity_int = raw_data[i+2]

        if polarity_int == "-1":
            polarity = "negative"
        elif polarity_int == "0":
            polarity = "neutral"
        elif polarity_int == "1":
            polarity = "positive"
        else:
            raise AttributeError(polarity_int)

        from_idx = sent_text_with_ph.find("$T$")
        to_idx = from_idx + len(term)
        sent_text = sent_text_with_ph.replace("$T$", term)

        sample = {
            "sent_id": str(i//3), "sent_text": sent_text,
            "terms": [{"term": term, "polarity": polarity, "from": from_idx, "to": to_idx}],
            "categories": [],
        }
        absa_dataset.append(sample)
    return absa_dataset

def read_psst_normal_dataset(data_path, offset):
    # str2id
    str2id = {}
    with open(os.path.join(data_path, "dictionary.txt"), encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if idx == 0:
                continue
            data = line.strip().split("|")
            assert len(data) == 2
            str2id[data[0]] = data[1]
    # id2label
    id2label = {}
    with open(os.path.join(data_path, "sentiment_labels.txt"), encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if idx == 0:
                continue
            data = line.strip().split("|")
            assert len(data) == 2
            val = float(data[1])
            # offset = 0.1
            if val < 0.5 - offset:
                label = 0
            elif val < 0.5 + offset:
                label = 1
            else:
                label = 2
            id2label[data[0]] = label

    absa_dataset = []
    for text, text_id in str2id.items():
        label = id2label[text_id]
        if label == 0:
            polarity = "negative"
        elif label == 1:
            polarity = "neutral"
        elif label == 2:
            polarity = "positive"
        else:
            raise AttributeError

        sample = {
            "sent_id": "psst-"+str(text_id), "sent_text": text,
            "terms": [{"term": None, "polarity": polarity, "from": -1, "to": -1}],
            "categories": [],
        }
        absa_dataset.append(sample)

    return absa_dataset



DATASET2PATH = {
    "rest14": os.path.join("./datasets/SemEval-14"),
    "lap14": os.path.join("./datasets/SemEval-14"),
    "rest15": os.path.join("./datasets/SemEval-15"),
    "rest16": os.path.join("./datasets/SemEval-16"),
    "twitter": os.path.join("./datasets/Twitter"),

    "rest14_wordnet_bias": os.path.join("./datasets/SemEval-14/bias_rest14"),
    "lap14_wordnet_bias": os.path.join("./datasets/SemEval-14/bias_lap14"),
    "rest15_wordnet_bias": os.path.join("./datasets/SemEval-15/bias_rest15"),
    "rest16_wordnet_bias": os.path.join("./datasets/SemEval-16/bias_rest16"),
    "twitter_wordnet_bias": os.path.join("./datasets/twitter/bias_twitter"),

    "lap14_wordnet_bias_distillation": os.path.join("./datasets/SemEval-14/bias_lap14"),
    "rest14_wordnet_bias_distillation": os.path.join("./datasets/SemEval-14/bias_rest14"),
    "rest15_wordnet_bias_distillation": os.path.join("./datasets/SemEval-15/bias_rest15"),
    "rest16_wordnet_bias_distillation": os.path.join("./datasets/SemEval-16/bias_rest16"),
    "twitter_wordnet_bias_distillation": os.path.join("./datasets/Twitter/bias_twitter"),

    "rest14_wordnet_bias_opinions": os.path.join("./datasets/SemEval-14/bias_rest14"),
    "lap14_wordnet_bias_opinions": os.path.join("./datasets/SemEval-14/bias_lap14"),
    "rest15_wordnet_bias_opinions": os.path.join("./datasets/SemEval-15/bias_rest15"),
    "rest16_wordnet_bias_opinions": os.path.join("./datasets/SemEval-16/bias_rest16"),

    "psst": os.path.join("./datasets/SST2-Data/stanfordSentimentTreebank"),
}

def get_raw_datasets(dataset, dataset_dir=None):
    if dataset == "rest14":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "Restaurants_Train_v2.xml")
        test_dataset_path = os.path.join(dataset_dir, "Restaurants_Test_Gold.xml")
        dev_dataset_path = test_dataset_path
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_semeval14_absa_xml(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
        # dev_dataset_raw = read_semeval14_absa_neutral_xml(dev_dataset_path)
        # test_dataset_raw = read_semeval14_absa_neutral_xml(test_dataset_path)   # explore the neutral results
    elif dataset == "lap14":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "Laptop_Train_v2.xml")
        test_dataset_path = os.path.join(dataset_dir, "Laptops_Test_Gold.xml")
        dev_dataset_path = test_dataset_path
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_semeval14_absa_xml(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
        # train_dataset_raw = read_semeval14_absa_neu_sentence_xml(train_dataset_path)
        # dev_dataset_raw = read_semeval14_absa_neutral_xml(dev_dataset_path)
        # test_dataset_raw = read_semeval14_absa_neutral_xml(test_dataset_path)
    elif dataset == "twitter":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "train.raw")
        test_dataset_path = os.path.join(dataset_dir, "test.raw")
        dev_dataset_path = test_dataset_path
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_twitter_absa_dataset(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
        # dev_dataset_raw = test_dataset_raw = read_twitter_absa_neutral_dataset(test_dataset_path)
    elif dataset in ["rest15", "rest16"]:
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "Restaurants_Train_Final.xml")
        test_dataset_path = os.path.join(dataset_dir, "Restaurants_Test.xml")
        dev_dataset_path = test_dataset_path
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_rest15_16_absa_xml(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
    elif dataset == "psst":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_raw = read_psst_normal_dataset(os.path.join(dataset_dir), offset=0.1)
        dev_dataset_raw = read_psst_normal_dataset(os.path.join(dataset_dir), offset=0.1)
        test_dataset_raw = read_psst_normal_dataset(os.path.join(dataset_dir), offset=0.1)
    elif dataset.split("_")[-1] == "bias":
        file_prefix = "_".join(dataset.split("_")[1:])
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, file_prefix + "_train.json")
        test_dataset_path = os.path.join(dataset_dir, file_prefix + "_test.json")
        dev_dataset_path = test_dataset_path
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_bias_absa_json(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
    elif dataset.split("_")[-1] == "distillation":
        file_prefix = "_".join(dataset.split("_")[1:])
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, file_prefix + "_train.json")
        test_dataset_path = os.path.join(dataset_dir, file_prefix + "_test.json")
        dev_dataset_path = test_dataset_path
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_bias_absa_json(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
    elif dataset.split("_")[-1] == "opinions":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        test_dataset_path = os.path.join(dataset_dir, dataset + "_test.json")
        train_dataset_raw, dev_dataset_raw = [], []
        test_dataset_raw = read_bias_absa_json(test_dataset_path)
    else:
        raise NotImplementedError(dataset, dataset_dir)
    return train_dataset_raw, dev_dataset_raw, test_dataset_raw


class AbsaDataset(Dataset):
    def __init__(
            self, config, absa_dataset, data_format, data_type, tokenizer_type, tokenizer, max_seq_length=64,
            concat_way="naive", nums_label=3, rate=1, nums_bias_label=3, max_span_width=15, *args, **kwargs):
        self.config = config

        self.absa_dataset = absa_dataset
        self.data_format = data_format
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.nums_label = nums_label
        self.nums_bias_lable = nums_bias_label
        self.max_span_width = max_span_width

        self.concat_way = concat_way
        self.dataset_path = None

        self.tokenizer_type = tokenizer_type

        self.standalone = True
        self.rm_conflict = True

        # begin to process
        self.example_list = []
        if data_format == "term" or data_format == "term_span":
            # with term as unit
            for sample in self.absa_dataset:
                for idx_term, term_dict in enumerate(sample["terms"]):
                    if self.rm_conflict and term_dict["polarity"] == "conflict":
                        continue
                    example = {
                        "sent_id": sample["sent_id"] + "-" + str(idx_term),
                        "sent_text": sample["sent_text"],
                        "terms": [term_dict],
                    }
                    self.example_list.append(example)
        elif data_format == "category":
            raise NotImplementedError
        elif data_format == "multi_aspects":
            for sample in self.absa_dataset:
                sample_aspects = []
                for idx_term, term_dict in enumerate(sample["terms"]):
                    if self.rm_conflict and term_dict["polarity"] == "conflict":
                        continue
                    sample_aspects.append(term_dict)
                if sample_aspects:
                    sample["terms"] = sample_aspects
                    self.example_list.append(sample)
        else:
            raise AttributeError(data_format)

        self.train_rate = rate
        self.example_list, _ = self.data_split(self.example_list, self.train_rate, shuffle=True)

        # Misc
        self.sep_token, self.cls_token, self.pad_token = \
            self.tokenizer.sep_token, self.tokenizer.cls_token, self.tokenizer.pad_token
        self.sep_id, self.cls_id, self.pad_id = self.tokenizer.convert_tokens_to_ids(
            [self.sep_token, self.cls_token, self.pad_token],
        )

    def data_split(self, full_list, ratio, shuffle=False):
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if n_total == offset:
            return full_list, []
        if shuffle:
            random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1, sublist_2

    def get_dep_distance(self, example, wpidx2tokenidx, num_wps, wp_list):
        # wpidx2tokenidx: 0 denotes the [CLS] and [SEP].

        sent_text = example["sent_text"]
        num_char = len(sent_text)
        doc = nlp(sent_text)
        nodes_list = doc.to_json()['tokens']  # each node is a dict.
        edges = []
        charidx2nodeidx = [-1 for _ in range(num_char)]
        for node in nodes_list:
            _start, _end = node['start'], node['end']
            # nodes_token.append(sent_text[_start : _end])
            edges.append((node['id'], node['head']))
            charidx2nodeidx[_start : _end] = [node['id'] for _ in range(_end - _start)]
            if _start - 1 >= 0 and charidx2nodeidx[_start-1] == -1:
                charidx2nodeidx[_start - 1] = node['id']

        # Omit special sentences which are hard to process.
        if sent_text == '乔布斯搞主题公园啦 ! -- steve jobs to help Disney turn stores into mini theme parks |' or \
            sent_text == '-LRB- iphone apps -RRB- - star walk が安くなっている - 5 stars astronomy guide -':
            asp_length = []
            for _ in example["terms"]:
                _tmp_list = torch.tensor([-19] * num_wps)
                asp_length.append(_tmp_list)
            return asp_length

        charidx2wpsidx = parse_tk_idx_list_wrt_char(self.tokenizer, sent_text, wp_list)


        # ------- Alignment the node and wordpiect to get new edges ------------
        if len(nodes_list) != num_wps:
            nodeidx2wpsidx = {}
            for _i, _ni in enumerate(charidx2nodeidx):
                if _ni not in nodeidx2wpsidx:
                    nodeidx2wpsidx[_ni] = set()
                nodeidx2wpsidx[_ni].add(charidx2wpsidx[_i])

            new_edges = set()
            for n1, n2 in edges:
                n1wps = list(nodeidx2wpsidx[n1])
                n2wps = list(nodeidx2wpsidx[n2])
                _tmp_edges = set([(x, y) for x in n1wps for y in n2wps])
                new_edges = new_edges.union(_tmp_edges)
            edges = list(new_edges)
        # ------------------ Get the dependency distance --------------------
        graph = nx.Graph(edges)
        asp_length = []
        # for each aspect term, get a dependency distance list whose length is num_wordpiece
        for term_dict in example["terms"]:
            from_idx = term_dict["from"]
            to_idx = term_dict["to"]
            # get the index of the aspect' start and end wordpiece
            wps_from_idx, wps_to_idx = char_to_token_span(charidx2wpsidx, from_idx, to_idx)
            length_dict_list = []
            # if there are more than 1 wordpiece in current aspect, record all distance
            for _term_token_idx in range(wps_from_idx, wps_to_idx):
                node_list = list(graph.nodes)
                length_dict_list.append(nx.shortest_path_length(graph, target=_term_token_idx))
            _tmp_list = [[] for _ in range(wps_to_idx - wps_from_idx)]

            for _i, length_dict in enumerate(length_dict_list):
                for _j in range(num_wps):
                    if _j in length_dict.keys():
                        _tmp_list[_i].append(length_dict[_j])
                    else:
                        _tmp_list[_i].append(19)

            _tmp_list = torch.min(torch.tensor(_tmp_list), dim=0).values

            assert len(_tmp_list) == num_wps
            asp_length.append(_tmp_list)
        return asp_length

    def str2ids(self, text, max_len=None):
        text = self.tokenizer.cls_token + " " + text
        wps = self.tokenizer.tokenize(text)
        if max_len is not None:
            wps = self.tokenizer.tokenize(text)[:max_len-1]
        wps.append(self.tokenizer.sep_token)
        return self.tokenizer.convert_tokens_to_ids(wps)

    def _get_label(self, polarity):
        if polarity == "negative":
            label = 0
        elif polarity == "neutral":
            label = 1
        elif polarity == "positive":
            if self.nums_label == 3:
                label = 2
            elif self.nums_label == 2:
                label = 0
        else:
            raise AttributeError(polarity)
        return label

    def _get_bias_label(self, bias):
        if bias == "negative":
            label = 0
        elif bias == "neutral":
            label = 1
        elif bias == "positive":
            if self.nums_bias_lable == 3:
                label = 2
            elif self.nums_bias_lable == 2:
                label = 0
        elif bias == "non-neutral":
            label = 0
        else:
            raise AttributeError(bias)
        return label

    def __getitem__(self, item):
        example = self.example_list[item]
        sent_text = example["sent_text"]
        if self.concat_way == "naive": # the concatenation of aspect ans sentence
            aspect_term = example["terms"][0]["term"]  # each example["terms"] is a list containing only one term-dict
            # since we have splited the list in __init__
            polarity = example["terms"][0]["polarity"]
            aspect_term_ids = self.str2ids(aspect_term, max_len=16)
            sent_ids = self.str2ids(sent_text, max_len=self.max_seq_length-len(aspect_term_ids)+1)
            input_ids = aspect_term_ids + sent_ids[1:]   # Begin from 1 to remove cls_token id.
            segment_ids = [0]*len(aspect_term_ids) + [1]*len(sent_ids[1:])  # Aspect_term is in front of the sentence
        elif self.concat_way == "sent":  # sentence-level sentiment classification
            polarity = example["terms"][0]["polarity"]
            input_ids = self.str2ids(sent_text, max_len=self.max_seq_length)
            segment_ids = [0]*len(input_ids)
        elif self.concat_way == "multi_aspects1":  # MLP
            sent_tokens = split_to_token(self.tokenizer, sent_text)
            charidx2tokenidx = parse_tk_idx_list_wrt_char(self.tokenizer, sent_text, sent_tokens)
            wp_list, id_list, wpidx2tokenidx = continue_tokenize_for_wordpiece(self.tokenizer, sent_tokens)
            aspects_pos = []
            multi_labels = []
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
                aspects_pos.extend([wp_from_idx, wp_to_idx])   # save the aspects' position
                for pos_idx in range(wp_from_idx, wp_to_idx):
                    term_mask[pos_idx] = 1
                # label
                label = self._get_label(term_dict["polarity"])
                multi_labels.append(label)
                # if len(sent_tokens)  != len(id_list):
                #     token = []
            input_ids = [self.cls_id] + id_list + [self.sep_id]
            segment_ids = [0] + term_mask + [0]
            mask_ids = [1] * len(input_ids)

            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask_ids, dtype=torch.long), \
                   torch.tensor(segment_ids, dtype=torch.long), torch.tensor(multi_labels, dtype=torch.long), \
                   torch.tensor(aspects_pos, dtype=torch.long)
        elif self.concat_way == "adv_distillation":  # SARL

            example["sent_text"] = sent_text.replace('’','\'')  # '’' will make a difficult for the alignment between tokenizer and parser
            sent_text = example["sent_text"]
            sent_tokens = split_to_token(self.tokenizer, sent_text)
            charidx2tokenidx = parse_tk_idx_list_wrt_char(self.tokenizer, sent_text, sent_tokens)
            wp_list, id_list, wpidx2tokenidx = continue_tokenize_for_wordpiece(self.tokenizer, sent_tokens)
            aspects_pos = []
            multi_labels = []
            multi_bias_labels = []
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
                # label
                label = self._get_label(term_dict["polarity"])
                bias_label = self._get_bias_label(term_dict["bias"])
                multi_labels.append(label)
                multi_bias_labels.append(bias_label)

            input_ids = [self.cls_id] + id_list + [self.sep_id]
            segment_ids = [0] + term_mask + [0]
            mask_ids = [1] * len(input_ids)

            # provide the sentiment score and wpidx2tokenidx list in train phrase;
            wpidx2tokenidx = torch.tensor(wpidx2tokenidx, dtype=torch.long)
            with open(os.path.join("./datasets/SentiWords.json")) as f:
                senti_dict = json.load(f)
            sentiment_score = [0 for _ in range(len(id_list))]
            sentiment_score = torch.tensor(sentiment_score, dtype=torch.float)
            for _token_id, token in enumerate(sent_tokens):
                token = token[1:] if token[0] == ' ' else token
                token = wnl.lemmatize(token)
                if token in senti_dict:
                    sentiment_score[torch.squeeze((wpidx2tokenidx == _token_id).nonzero())] = abs(senti_dict[token])

            # prepare for providing wpidx2tokenidx
            wpidx2tokenidx = torch.cat([torch.tensor([-1]), wpidx2tokenidx, torch.tensor([-1])], dim=0) + 1
            assert len(input_ids) == len(wpidx2tokenidx)

            # get the dependency distance between each wordpiece and each aspect, then return a [num_asp, num_wps] tensor
            asp_length = self.get_dep_distance(example, wpidx2tokenidx, len(input_ids) - 2, wp_list)

            input_ids = torch.tensor(input_ids, dtype=torch.long)

            spans_list = example["spans"]
            starts, ends, senti_distribution = [], [], []
            num_spans = len(spans_list)
            for _i in range(num_spans):
                starts.append(spans_list[_i][0])
                ends.append(spans_list[_i][1])
                senti_distribution.append(spans_list[_i][2])

            senti_distribution = torch.tensor(senti_distribution, dtype=torch.float32)

            return input_ids, torch.tensor(mask_ids, dtype=torch.long), \
                   torch.tensor(segment_ids, dtype=torch.long), torch.tensor(multi_labels, dtype=torch.long), \
                   torch.tensor(aspects_pos, dtype=torch.long), \
                   sentiment_score, wpidx2tokenidx, asp_length, torch.tensor(multi_bias_labels, dtype=torch.long), \
                   torch.tensor(starts, dtype=torch.long), torch.tensor(ends, dtype=torch.long), senti_distribution
        else:
            raise NotImplementedError
        mask_ids = [1] * len(input_ids)

        # label
        if polarity == "negative":
            label = 0
        elif polarity == "neutral":
            label = 1
        elif polarity == "positive":
            if self.nums_label == 3:
                label = 2
            elif self.nums_label == 2:
                label = 0
        elif polarity == -1:
            label = -1
        else:
            raise AttributeError(polarity)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask_ids, dtype=torch.long), \
               torch.tensor(segment_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long),

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch))  # 5
        return_list = []
        for _idx_t, _tensors in enumerate(tensors_list):
            if self.data_format == 'term_hinge':
                if _idx_t == 0:
                    padding_value = self.pad_id
                elif _idx_t < 3:
                    padding_value = 0
                elif _idx_t >= 3:
                    padding_value = -1
            else:
                if _idx_t == 0:
                    padding_value = self.pad_id
                else:
                    padding_value = 0
            _tensors = [_t.t() for _t in _tensors]
            if _tensors[0].dim() >= 1:
                _tensors = [_t.t() for _t in _tensors]
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(
                        _tensors, batch_first=True, padding_value=padding_value),  # .transpose(-1, -2)
                )
            else:
                return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list)

    @classmethod
    def batch2feed_dict(cls, batch):
        inputs = {
            'input_ids': batch[0],  # bs, sl
            'attention_mask': batch[1],  #
            'token_type_ids': batch[2],  #
            "labels": batch[-1],  #
        }
        return inputs


    def data_collate_fn_multi(self, batch):
        tensors_list = list(zip(*batch)) #  the num of keys of input-dict
        return_list = []
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t == 0 or _idx_t == 11:
                padding_value = self.pad_id
            elif _idx_t < 3 or _idx_t > 11 :
                padding_value = 0
            else:
                padding_value = -1

            if  _idx_t == 7 or _idx_t > 10:
                _tensors = [_t for _t in _tensors]
            else:
                _tensors = [_t.t() for _t in _tensors]
            if _idx_t == 7 or _idx_t > 10:
                # get max seq len for padding
                _max_len_last_dim = 0
                for _tensor in _tensors:
                    _local_max_len_last_dim = max(len(_t) for _t in list(_tensor))
                    _max_len_last_dim = max(_max_len_last_dim, _local_max_len_last_dim)
                # 2d padding
                _new_tensors = []
                for _tensor in _tensors:  # _tensor: (seq_len, seq_len)
                    inner_tensors = []
                    for idx, _ in enumerate(list(_tensor)):
                        _pad_shape = _max_len_last_dim - len(_tensor[idx])
                        _pad_tensor = torch.tensor([padding_value] * _pad_shape, device=_tensor[idx].device, dtype=_tensor[idx].dtype)
                        _new_inner_tensor = torch.cat([_tensor[idx], _pad_tensor], dim=0)  # padding_len
                        inner_tensors.append(_new_inner_tensor)
                    _tensors_tuple = tuple(ts for ts in inner_tensors)
                    _new_tensors.append(torch.stack(_tensors_tuple, dim=0))
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(_new_tensors, batch_first=True, padding_value=padding_value),
                )
            else:
                return_list.append(
                        torch.nn.utils.rnn.pad_sequence(
                            _tensors, batch_first=True, padding_value=padding_value),
                    )


        return tuple(return_list)

    @classmethod
    def batch2feed_dict_multi(cls, batch, concat_way=None):

        if concat_way == 'adv_distillation':
            inputs = {
                'input_ids': batch[0],  # bs, sl
                'attention_mask': batch[1],  #
                'token_type_ids': batch[2],  #
                "multi_labels": batch[3],  #
                "multi_aspects_pos": batch[4],
                "sentiment_scores": batch[5],
                "wps2tokens": batch[6],
                "dep_distances":batch[7],
                "multi_bias_labels": batch[8],
                "starts_position":batch[9],
                "ends_position":batch[10],
                "senti_distributions": batch[11]
            }
        else:
            inputs = {
                'input_ids': batch[0],  # bs, sl
                'attention_mask': batch[1],  #
                'token_type_ids': batch[2],  #
                "multi_labels": batch[3],  #
                "multi_aspects_pos":batch[-1]
            }
        return inputs

    def __len__(self):
        return len(self.example_list)









