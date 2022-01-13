import sys
import regex as re
from transformers import BertTokenizer, RobertaTokenizer

def split_to_token(tokenizer, sent_text):
    tgt_special_tokens = tokenizer.all_special_tokens   # BERT <class 'list'>: ['[SEP]','[UNK]','[MASK]','[CLS]','[PAD]']
                                                        # RoBERTa <class 'list'>: ['<mask>', '<unk>', '<pad>', '<s>', '</s>']
    def split_on_token(tok, text):
        result = []
        split_text = text.split(tok)
        for i, sub_text in enumerate(split_text):
            sub_text = sub_text.strip()
            if i == 0 and not sub_text:
                result += [tok]
            elif i == len(split_text) - 1:
                if sub_text:
                    result += [sub_text]
                else:
                    pass
            else:
                if sub_text:
                    result += [sub_text]
                result += [tok]
        return result

    def split_on_tokens(tok_list, text, tokenize_fn):
        if not text:
            return []
        if not tok_list:
            return re.findall(tokenizer.pat, sent_text)

        tokenized_text = []
        text_list = [text]
        for tok in tok_list:
            tokenized_text = []
            for sub_text in text_list:
                if sub_text not in tgt_special_tokens:
                    tokenized_text += split_on_token(tok, sub_text)
                else:
                    tokenized_text += [sub_text]
            text_list = tokenized_text

        return sum((tokenize_fn(token) if token not in tgt_special_tokens \
                        else [token] for token in tokenized_text), [])

    if isinstance(tokenizer, BertTokenizer):
        bert_tokenzie_fn = tokenizer.basic_tokenizer.tokenize   # split text into full words
        sent_token_list = split_on_tokens(tgt_special_tokens, sent_text, bert_tokenzie_fn)
    elif isinstance(tokenizer, RobertaTokenizer):
        roberta_tokenize_fn = lambda _arg: re.findall(tokenizer.pat, _arg)
        sent_token_list = split_on_tokens(tgt_special_tokens, sent_text, roberta_tokenize_fn)
    else:
        raise NotImplementedError

    return sent_token_list

def parse_tk_idx_list_wrt_char(tokenizer, _text, _token_list):
    span_list = []
    if isinstance(tokenizer, BertTokenizer):
        if tokenizer.basic_tokenizer.do_lower_case:
            _text = _text.lower()

        _tmp_find_idx = 0
        for _idx, _tk in enumerate(_token_list):
            if '##' in _tk:
                _tk = _tk.replace('##', '')

            _found_idx = _text.find(_tk, _tmp_find_idx)
            if _idx == 0 and _found_idx != 0:
                _found_idx = 0
            _end_idx = _found_idx + len(_tk)
            if _found_idx < 0:
                for _c in _tk:
                    print(_c + '_' + str(ord(_c)))
            assert _found_idx >= 0
            # else:
            span_list.append(_found_idx)
            _tmp_find_idx = _end_idx
        span_list.append(len(_text))
    elif isinstance(tokenizer, RobertaTokenizer):
        _tmp_find_idx = 0
        for _idx, _tk in enumerate(_token_list):
            if _text == '-LRB- ° ё ° -RRB- RT fubiz Amazing ipad Animations .':
                if _tk == 'Ġ':
                    _tk = ''
                if _tk == 'Ñ':
                    _tk = ' ё'
                if _tk == 'ĳ':
                    _tk = ''
                if _tk == 'ĠÂ°':
                    _tk = ' °'
            if _text == 'reaaaaalllly love your acting in harry potter ! Why do n\'t you come to Korea ? ㅠㅠ':
                if _tk == 'ã':
                    _tk = 'ㅠ'
                if _tk == 'ħ':
                    _tk = ''
                if _tk == 'ł':
                    _tk = ''
            # else:
            if _tk != '':
                if _tk[0] == 'Ġ':
                    _tk = ' ' + _tk[1:]
                if _tk[0] == 'Â':
                    _tk = chr(160)
                # continue
            # for twitter
            # if _tk == 'cÃ©':
            #     _tk = 'cé'
            if _tk == ' âĻ':
                _tk = ' '
            if _tk == '¥':
                _tk = chr(9829)
            if _tk == ' âĺ':
                _tk = ' '
            if _tk == 'ĳ':
                _tk = chr(9745)
            if _tk =='«':
                _tk = '♫'
            if _tk == ' Â':
                _tk = ' '
            if _tk == 'º':
                _tk = '☺'
            if 'Ã«' in _tk:
                _tk = _tk.replace('Ã«', 'ë')
            if _tk == ' âĺħ':
                _tk = ' ★'
            if _tk == ' âĢ¢':
                _tk = ' •'
            if _tk == 'âĺ':
                _tk = ''
            # if _tk == 'Ã±':
            #     _tk = 'ñ'
            if _tk == 'âĻ¥':
                _tk = '♥'
            if _tk == ' âľĶ':
                _tk = ' ✔'
            if 'Ãł' in _tk:
                _tk = _tk.replace('Ãł', 'à')
            if 'Ã©' in _tk:
                _tk = _tk.replace('Ã©', 'é')
            if 'Â®' in _tk:
                _tk = _tk.replace('Â®', '®')
            if _tk == '®':
                _tk = '☮'
            if _tk == 'ª':
                _tk = '♪'
            if _tk == ' â':
                _tk = ''
            if _tk == 'ŀ':
                _tk = ''
            if _tk == 'ľ':
                _tk = '➜'
            if 'Ãĸ' in _tk:
                _tk = _tk.replace('Ãĸ', 'Ö')
            if 'Ã³' in _tk:
                _tk = _tk.replace('Ã³', 'ó')
            if 'ÃŃ' in _tk:
                _tk = _tk.replace('ÃŃ', 'í')
            if _tk == 'ł':
                _tk = '☠'
            if _tk == 'âĻ':
                _tk = ''
            if 'Ã£' in _tk:
                _tk = _tk.replace('Ã£', 'ã')
            if _tk == 'âľ':
                _tk = ''
            if _tk == 'Ķ':
                _tk = '✔'
            if 'Ãª' in _tk:
                _tk = _tk.replace('Ãª', 'ê')
            if _tk == ' ãģ':
                _tk = 'っ'
            if _tk == '£':
                _tk = ''
            if _tk == 'ãģĭ':
                _tk = 'か'
            if _tk == 'ãĢĤ':
                _tk = '。'
            if _tk == '¤':
                _tk = ' ➤'
            if 'Ã¡' in _tk:
                _tk = _tk.replace('Ã¡', 'á')
            if _tk == ' Ã':
                _tk = ' '
            if '§' in _tk:
                _tk = _tk.replace('§', 'ç')
            if _tk == '²':
                _tk = ' ➲'
            if 'Å¡' in _tk:
                _tk = _tk.replace('Å¡', 'š')
            if _tk == ' Â°':
                _tk = ' °'
            # if  in _tk:
            #     _tk = _tk.replace('Ã§', 'ç')
            if _tk == 'Ã§a':
                _tk = 'ça'
            if _tk == 'Ãça':
                _tk = 'ça'
            if _tk == 'Ãç':
                _tk = 'ç'

            if 'Ã±' in _tk:
                _tk = _tk.replace('Ã±', 'ñ')
            if _tk == '¹':
                _tk = '☹'
            if _tk == ' MÃ¼':
                _tk = 'Mü'

            # for rest15
            if _tk == ' âĢĵ':
                _tk = ' –'
            if _tk == 'âĢ':
                _tk = '’'
            if _tk == 'Ļ':
                _tk = ''
            if _tk == ' âĢ':
                _tk = ' ‘'
            if _tk == 'ĺ':
                _tk = ''

            # for rest16
            if _tk == 'âĢĵ':
                _tk = '–'
            if _tk == 'âĢ¦':
                _tk = '…'



            _found_idx = _text.find(_tk, _tmp_find_idx)
            if _idx == 0 and _found_idx != 0:
                _found_idx = 0
            _end_idx = _found_idx + len(_tk)
            if _found_idx < 0:
                for _c in _tk:
                    print(_c + '_' + str(ord(_c)))
            assert _found_idx >= 0
            # else:
            span_list.append(_found_idx)
            _tmp_find_idx = _end_idx
        span_list.append(len(_text))
    else:
        raise NotImplementedError
    if len(span_list) != len(_token_list) + 1:
        print(_text, _token_list)
        print(span_list)
    assert len(span_list) == len(_token_list) + 1
    # # build char2token
    tk_idx_list = []
    for _idx_tk, _char_start in enumerate(span_list[:-1]):
        _char_end = span_list[_idx_tk + 1]
        _token_act_len = _char_end - _char_start
        tk_idx_list.extend([_idx_tk] * _token_act_len)
    return tk_idx_list

def continue_tokenize_for_wordpiece(tokenizer, token_list):
    wp_list = []
    pos_list = []
    if isinstance(tokenizer, BertTokenizer):
        for _idx_tk, _tk in enumerate(token_list):
            _wps = tokenizer.tokenize(_tk)
            wp_list.extend(_wps)
            pos_list.extend([_idx_tk] * len(_wps))
    elif isinstance(tokenizer, RobertaTokenizer):
        for _idx_tk, _tk in enumerate(token_list):
            if _tk in tokenizer.all_special_tokens:
                _wps = [_tk]
            else:
                if sys.version_info[0] == 2:
                    _tk = ''.join(tokenizer.byte_encoder[ord(b)] for b in _tk)
                else:
                    _tk = ''.join(tokenizer.byte_encoder[b] for b in _tk.encode('utf-8'))
                _wps = [bpe_token for bpe_token in tokenizer.bpe(_tk).split(' ')]
            wp_list.extend(_wps)
            pos_list.extend([_idx_tk] * len(_wps))
    else:
        raise NotImplementedError

    id_list = tokenizer.convert_tokens_to_ids(wp_list)
    assert len(wp_list) == len(id_list)
    return wp_list, id_list, pos_list

def char_to_token_span(charidx2tokenidx, char_from, char_to):
    token_from = charidx2tokenidx[char_from]
    token_to = charidx2tokenidx[char_to-1] + 1
    return token_from, token_to

