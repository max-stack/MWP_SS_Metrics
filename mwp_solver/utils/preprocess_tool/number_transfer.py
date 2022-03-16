import re
import json
import random
from copy import deepcopy
from collections import OrderedDict

import nltk
import stanza

from utils.utils import read_json_data, str2float, lists2dict
from utils.enum_type import MaskSymbol, NumMask, SpecialTokens, EPT, TaskType
from utils.data_structure import DependencyTree
from utils.preprocess_tool.number_operator import english_word_2_num


def number_transfer(datas, task_type, mask_type, min_generate_keep,linear_dataset, equ_split_symbol=';',vocab_level='word'):
    """number transfer
    Args:
        datas (list): dataset.
        dataset_name (str): dataset name.
        task_type (str): [single_equation | multi_equation], task type.
        min_generate_keep (int): generate number that count greater than the value, will be kept in output symbols.
        equ_split_symbol (str): equation split symbol, in multiple-equation dataset, symbol to split equations, this symbol will be repalced with special token SpecialTokens.BRG.
    
    Returns:
        tuple(list,list,int,list):
        processed datas, generate number list, copy number, unk symbol list.
    """
    transfer = number_transfer_helper
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    processed_datas = []
    unk_symbol = []
    for data in datas:
        if task_type == TaskType.SingleEquation:
            new_data = transfer(data, mask_type, linear_dataset, vocab_level)
        elif task_type == TaskType.MultiEquation:
            new_data = transfer(data, mask_type, equ_split_symbol,vocab_level)
        else:
            raise NotImplementedError
        num_list = new_data["number list"]
        out_seq = new_data["equation"]
        copy_num = len(new_data["number list"])

        for idx, s in enumerate(out_seq):
            # tag the num which is generated
            if s[0] == '-' and len(s) >= 2 and s[1].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s[0].isdigit() and s not in generate_nums and s not in num_list:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in num_list:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        if copy_num > copy_nums:
            copy_nums = copy_num

        # get unknown number
        if task_type == TaskType.SingleEquation:
            if linear_dataset:
                for s in out_seq:
                    if len(s) == 1 and s.isalpha():
                        if s in unk_symbol:
                            continue
                        else:
                            unk_symbol.append(s)
            else:
                pass
        elif task_type == TaskType.MultiEquation:
            for s in out_seq:
                if len(s) == 1 and s.isalpha():
                    if s in unk_symbol:
                        continue
                    else:
                        unk_symbol.append(s)
        else:
            raise NotImplementedError

        processed_datas.append(new_data)
    # keep generate number
    generate_number = []
    for g in generate_nums:
        if generate_nums_dict[g] >= min_generate_keep:
            generate_number.append(g)
    return processed_datas, generate_number, copy_nums, unk_symbol

def seg_and_tag(st, nums_fraction, nums):  # seg the equation and tag the num
    res = []
    pos_st = re.search(r"([+]|-|[*]|/|[(]|=)-((\d+\.?\d*))", st)  #search negative number but filtate minus symbol
    if pos_st:
        p_start = pos_st.start() + 1
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
        if p_end < len(st):
            res += seg_and_tag(st[p_end:], nums_fraction, nums)
        return res
    for n in nums_fraction:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag(st[:p_start], nums_fraction, nums)
            try:
                res.append(nums[n])
            except:
                res.append(n)
            if p_end < len(st):
                res += seg_and_tag(st[p_end:], nums_fraction, nums)
            return res
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)  #search number including number with % symbol
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag(st[:p_start], nums_fraction, nums)
        st_num = st[p_start:p_end]
        try:
            res.append(nums[st_num])
        except:
            number = str(str2float(st_num))
            try:
                if abs(eval(number) - eval(st_num)) < 1e-4:
                    res.append(nums[number])
                else:
                    res.append(number)
            except:
                res.append(number)
        if p_end < len(st):
            res += seg_and_tag(st[p_end:], nums_fraction, nums)
        return res
    for ss in st:
        if ss.isalpha():
            res.append(ss.lower())
        elif ss == " ":
            continue
        else:
            res.append(ss)
    return res

def number_transfer_helper(data, mask_type,linear,vocab_level='word'):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")
    
    seg = data["original_text"].split(" ")
    equations = data["equation"]
    equations = re.sub(r"[a-zA-Z]{2,}", "x", equations)

    # match and split number
    input_seq = []
    for s in seg:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            input_seq.append(str(str2float(s[pos.start():pos.end()])))
            if pos.end() < len(s):
                if vocab_level == 'char':
                    input_seq += [c for c in s[pos.end():]]
                else:
                    input_seq.append(s[pos.end():])
        else:
            if s == '':
                continue
            if vocab_level == 'char':
                input_seq += [c for c in s]
            else:
                input_seq.append(s)
   
    input_seq, num_list, num_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction = get_num_pos(input_seq, mask_type, pattern)

    out_seq = seg_and_tag(equations, nums_fraction, nums)

    source = deepcopy(input_seq)
    for pos in all_pos:
        for key, value in num_pos_dict.items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)

    assert len(num_list) == len(num_pos)

    #copy data
    new_data = data
    new_data["question"] = input_seq
    new_data["equation"] = out_seq
    new_data["ques source 1"] = source
    new_data["number list"] = num_list
    new_data["number position"] = num_pos
    return new_data

def get_num_pos(input_seq, mask_type, pattern):
    if mask_type == MaskSymbol.NUM:
        sent_mask_list = NumMask.NUM
        equ_mask_list = NumMask.number
    elif mask_type == MaskSymbol.alphabet:
        sent_mask_list = NumMask.alphabet
        equ_mask_list = NumMask.alphabet
    elif mask_type == MaskSymbol.number:
        sent_mask_list = NumMask.number
        equ_mask_list = NumMask.number
    nums = OrderedDict()
    num_list = []
    num_pos = []
    num_pos_dict = {}

    if mask_type == MaskSymbol.NUM:
        # find all number position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                num_list.append(word)
                num_pos.append(word_pos)
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_pos_dict[word] = [word_pos]

        mask_list = equ_mask_list[:len(num_list)]
        new_num_list = []
        new_mask_list = []
        for i in num_list:
            if num_list.count(i) != 1:
                x = 1
            if num_list.count(i) == 1:
                new_num_list.append(i)
                new_mask_list.append(mask_list[num_list.index(i)])
            else:
                pass
        nums = lists2dict(new_num_list, new_mask_list)
    else:
        # find all number position
        for word_pos, word in enumerate(input_seq):
            pos = re.search(pattern, word)
            if pos and pos.start() == 0:
                if word in num_pos_dict:
                    num_pos_dict[word].append(word_pos)
                else:
                    num_list.append(word)
                    num_pos_dict[word] = [word_pos]
        num_list = sorted(num_list, key=lambda x: max(num_pos_dict[x]), reverse=False)
        nums = lists2dict(num_list, equ_mask_list[:len(num_list)])

    nums_for_ques = lists2dict(num_list, sent_mask_list[:len(num_list)])

    # all number position
    all_pos = []
    if mask_type == MaskSymbol.NUM:
        all_pos = deepcopy(num_pos)
    else:
        for num, mask in nums_for_ques.items():
            for pos in num_pos_dict[num]:
                all_pos.append(pos)

    # final numbor position
    final_pos = []
    if mask_type == MaskSymbol.NUM:
        final_pos = deepcopy(num_pos)
    else:
        for num in num_list:
            # select the latest position as the number position
            # if the number corresponds multiple positions
            final_pos.append(max(num_pos_dict[num]))

    # number transform
    for num, mask in nums_for_ques.items():
        for pos in num_pos_dict[num]:
            input_seq[pos] = mask

    # nums_fraction = []
    # for num, mask in nums.items():
    #     if re.search("\data*\(\data+/\data+\)\data*", num):
    #         nums_fraction.append(num)
    # nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
    nums_fraction = []
    for num, mask in nums.items():
        if re.search("\d*\(\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

    return input_seq, num_list, final_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction