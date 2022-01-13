import json
from tqdm import tqdm
from absa.absa_dataset import read_bias_absa_json
import os


def get_dataset(dataset, type="test"):
    num_sentences = 0
    raw_opinion_dataset_dir = os.path.join("./datasets/ASTE-Data-V2-EMNLP2020/")
    if type == "test":
        with open(raw_opinion_dataset_dir + dataset + '/test_triplets.txt', 'r', encoding='utf-8') as f:
            lines_test = f.readlines()
        lines = lines_test
    else:
        raise NotImplementedError


    polarity_map = {'N': 0, 'NEU': 'neutral', 'NEG': 'negative', 'POS': 'positive'}  # NO_RELATION is 0
    all_data = []
    total_num, pos, neu, neg = 0, 0, 0, 0
    for i in range(len(lines)):
        num_sentences += 1
        text, pairs = lines[i].strip().split('####')
        aspects_list = []
        tokens_list = text.split(' ')
        aspect_dict = {}
        last_asp_beg, last_asp_end = -1, -1
        for pair in eval(pairs):

            ap_beg, ap_end = pair[0][0], pair[0][-1] + 1
            asp_term = ' '.join(tokens_list[ap_beg:ap_end])

            if aspect_dict == {} or aspect_dict['term'] != asp_term:
                assert aspect_dict == {} or len(aspect_dict['polarity']) == 1

                aspect_dict = {}
                aspect_dict['term'] = asp_term
                aspect_dict['opinions'] = []
                aspect_dict['polarity'] = set()
                total_num += 1

            op_beg, op_end = pair[1][0], pair[1][-1] + 1
            aspect_dict['opinions'].append((' '.join(tokens_list[op_beg:op_end]), op_beg, op_end))

            polarity = polarity_map[pair[2]]
            aspect_dict['polarity'].add(polarity)
            if polarity == 'neutral':
                neu += 1
            elif polarity == 'positive':
                pos += 1
            elif polarity == 'negative':
                neg += 1

            if ap_beg == last_asp_beg and ap_end == last_asp_end:
                pass
            else:
                last_asp_beg, last_asp_end = ap_beg, ap_end
                aspects_list.append(aspect_dict)

        data = {
            'sent_text': text,
            'terms': aspects_list
        }
        all_data.append(data)

    return all_data

if __name__ == '__main__':

    dataset_list = ["lap14", "rest14", "rest15", "rest16"]
    for dataset in dataset_list:
        opinion_dataset = get_dataset(dataset)

        dataset_dir = "./datasets/" +"SemEval-" + dataset[-2:]
        normal_dataset = read_bias_absa_json(dataset_dir + "/bias_" + dataset + "/wordnet_bias_test.json")

        for _o_idx, o_sentence in enumerate(tqdm(opinion_dataset)):
            o_text = o_sentence['sent_text']
            o_aspects = o_sentence['terms']
            o_text_tokens = o_text.split(' ')
            flag = False
            for n_sentence in normal_dataset:
                n_text = n_sentence['sent_text']
                n_aspects = n_sentence['terms']

                if n_text.replace(' ','') == o_text.replace(' ',''):
                    flag = True

                    for o_asp in o_aspects:
                        for n_asp in n_aspects:
                            if o_asp['term'].replace(' ','') == n_asp['term'].replace(' ',''):
                                opinions = []
                                for opinion in o_asp['opinions']:
                                    op_term, op_beg, op_end = opinion

                                    if ' n\'t' in op_term:
                                        op_term = op_term.replace(' n\'t', 'n\'t')
                                    if op_term == 'o.k. ,':
                                        op_term = 'o.k.'
                                    if 'can not' in op_term:
                                        op_term = op_term.replace('can not', 'cannot')
                                    if op_term == 'WORST .':
                                        op_term = 'WORST'
                                    if op_term == 'LARGE .':
                                        op_term = 'LARGE'
                                    if op_term == 'GREAT !':
                                        op_term = 'GREAT!'
                                    if op_term == 'O.K .':
                                        op_term = 'O.K.'
                                    if op_term == '( A+++ )':
                                        op_term = '(A+++)'
                                    if op_term == 'COMPLAINT :':
                                        op_term = 'COMPLAINT:'

                                    mini_num_before_char = 0
                                    for i in range(op_beg):
                                        mini_num_before_char += len(o_text_tokens[i])
                                    op_from = n_text.find(op_term, mini_num_before_char)
                                    op_to = op_from + len(op_term)
                                    opinion_dict = {'op_term':op_term, 'op_from':op_from, 'op_to':op_to}
                                    opinions.append(opinion_dict)

                                n_asp['opinions'] = opinions

                    break

        with open(dataset_dir + "/bias_" + dataset + '/' + dataset + '_wordnet_bias_opinions_test.json', 'w') as f:
            json.dump(normal_dataset, f)

        print(dataset + " finished. ")




