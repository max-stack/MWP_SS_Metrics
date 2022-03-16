import re
import json
import random
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, List, Union

import nltk
import stanza
from word2number import w2n

from utils.utils import read_json_data, str2float, lists2dict
from utils.enum_type import MaskSymbol, NumMask, SpecialTokens, EPT


def id_reedit(trainset, validset, testset):
    r"""if some datas of a dataset hava the same id, re-edit the id for differentiate them. 

    example: There are two datas have the same id 709356. Make one of them be 709356 and the other be 709356-1.
    """
    id_dict = {}
    for data in trainset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in validset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in testset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in trainset:
        old_id = data['id']
        if id_dict[old_id] > 1:
            new_id = old_id + '-' + str(id_dict[old_id] - 1)
            data['id'] = new_id
            id_dict[old_id] = id_dict[old_id] - 1
    for data in validset:
        old_id = data['id']
        if id_dict[old_id] > 1:
            new_id = old_id + '-' + str(id_dict[old_id] - 1)
            data['id'] = new_id
            id_dict[old_id] = id_dict[old_id] - 1
    for data in testset:
        old_id = data['id']
        if id_dict[old_id] > 1:
            new_id = old_id + '-' + str(id_dict[old_id] - 1)
            data['id'] = new_id
            id_dict[old_id] = id_dict[old_id] - 1
    return trainset, validset, testset


def dataset_drop_duplication(trainset, validset, testset):
    id_dict = {}
    for data in trainset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in validset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    for data in testset:
        if not isinstance(data['id'], str):
            data['id'] = str(data['id'])
        try:
            id_dict[data['id']] = id_dict[data['id']] + 1
        except:
            id_dict[data['id']] = 1
    drop_train=[]
    drop_valid=[]
    drop_test=[]
    for idx,data in enumerate(trainset):
        old_id = data['id']
        if id_dict[old_id] > 1:
            drop_train.append(idx-len(drop_train))
            id_dict[old_id] = id_dict[old_id] - 1
    for idx,data in enumerate(validset):
        old_id = data['id']
        if id_dict[old_id] > 1:
            drop_valid.append(idx-len(drop_valid))
            id_dict[old_id] = id_dict[old_id] - 1
    for idx,data in enumerate(testset):
        old_id = data['id']
        if id_dict[old_id] > 1:
            drop_test.append(idx-len(drop_test))
            id_dict[old_id] = id_dict[old_id] - 1
    for idx in drop_train:
        trainset.pop(idx)
    for idx in drop_valid:
        validset.pop(idx)
    for idx in drop_test:
        testset.pop(idx)
    return trainset, validset, testset

def find_ept_numbers_in_text(text: str, append_number_token: bool = False):
    
    numbers = []
    new_text = []

    # Replace "[NON-DIGIT][SPACEs].[DIGIT]" with "[NON-DIGIT][SPACEs]0.[DIGIT]"
    text = re.sub("([^\\d.,]+\\s*)(\\.\\d+)", "\\g<1>0\\g<2>", text)
    # Replace space between digits or digit and special characters like ',.' with "⌒" (to preserve original token id)
    text = re.sub("(\\d+)\\s+(\\.\\d+|,\\d{3}|\\d{3})", "\\1⌒\\2", text)

    # Original token index
    i = 0
    prev_token = None
    for token in text.split(' '):
        # Increase token id and record original token indices
        token_index = [i + j for j in range(token.count('⌒') + 1)]
        i = max(token_index) + 1

        # First, find the number patterns in the token
        token = token.replace('⌒', '')
        number_patterns = EPT.NUMBER_AND_FRACTION_PATTERN.findall(token)
        if number_patterns:
            for pattern in number_patterns:
                # Matched patterns, listed by order of occurrence.
                surface_form = pattern[0]
                surface_form = surface_form.replace(',', '')

                # Normalize the form: use the decimal point representation with 15-th position under the decimal point.
                is_fraction = '/' in surface_form
                value = eval(surface_form)
                if type(value) is float:
                    surface_form = EPT.FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % value)

                numbers.append(dict(token=token_index, value=surface_form,
                                    is_text=False, is_integer='.' not in surface_form,
                                    is_ordinal=False, is_fraction=is_fraction,
                                    is_single_multiple=False, is_combined_multiple=False))

            new_text.append(EPT.NUMBER_AND_FRACTION_PATTERN.sub(' \\1 %s ' % EPT.NUM_TOKEN, token))
        else:
            # If there is no numbers in the text, then find the textual numbers.
            # Append the token first.
            new_text.append(token)

            # Type indicator
            is_ordinal = False
            is_fraction = False
            is_single_multiple = False
            is_combined_multiple = False

            subtokens = re.split('[^a-zA-Z0-9]+', token.lower())  # Split hypen-concatnated tokens like twenty-three
            token_value = None
            for subtoken in subtokens:
                if not subtoken:
                    continue

                # convert to singular nouns
                for plural, singluar in EPT.PLURAL_FORMS:
                    if subtoken.endswith(plural):
                        subtoken = subtoken[:-len(plural)] + singluar
                        break

                if subtoken in EPT.NUMBER_READINGS:
                    if not token_value:
                        # If this is the first value in the token, then set it as it is.
                        token_value = EPT.NUMBER_READINGS[subtoken]

                        is_ordinal = subtoken[-2:] in ['rd', 'th']
                        is_single_multiple = subtoken in EPT.MULTIPLES

                        if is_ordinal and prev_token == 'a':
                            # Case like 'A third'
                            token_value = 1 / token_value
                    else:
                        # If a value was set before reading this subtoken,
                        # then treat it as multiples. (e.g. one-million, three-fifths, etc.)
                        followed_value = EPT.NUMBER_READINGS[subtoken]
                        is_single_multiple = False
                        is_ordinal = False

                        if followed_value >= 100 or subtoken == 'half':  # case of unit
                            token_value *= followed_value
                            is_combined_multiple = True
                        elif subtoken[-2:] in ['rd', 'th']:  # case of fractions
                            token_value /= followed_value
                            is_fraction = True
                        else:
                            token_value += followed_value

            # If a number is found.
            if token_value is not None:
                if type(token_value) is float:
                    surface_form = EPT.FOLLOWING_ZERO_PATTERN.sub('\\1', '%.15f' % token_value)
                else:
                    surface_form = str(token_value)

                numbers.append(dict(token=token_index, value=surface_form,
                                    is_text=True, is_integer='.' not in surface_form,
                                    is_ordinal=is_ordinal, is_fraction=is_fraction,
                                    is_single_multiple=is_single_multiple,
                                    is_combined_multiple=is_combined_multiple))
                new_text.append(EPT.NUM_TOKEN)

        prev_token = token

    if append_number_token:
        text = ' '.join(new_text)

    return text, numbers

def constant_number(const):
    """
    Converts number to constant symbol string (e.g. 'C_3').
    To avoid sympy's automatic simplification of operation over constants.

    :param Union[str,int,float,Expr] const: constant value to be converted.
    :return: (str) Constant symbol string represents given constant.
    """
    if type(const) is str:
        if const in ['C_pi', 'C_e', 'const_pi', 'const_e']:
            # Return pi, e as itself.
            return True, const.replace('const_', 'C_')

        # Otherwise, evaluate string and call this function with the evaluated number
        const = float(const.replace('C_', '').replace('const_', '').replace('_', '.'))
        return constant_number(const)
    elif type(const) is int or int(const) == float(const):
        # If the value is an integer, we trim the following zeros under decimal points.
        return const >= 0, 'C_%s' % int(abs(const))
    else:
        if abs(const - 3.14) < 1E-2:  # Including from 3.14
            return True, 'C_pi'
        if abs(const - 2.7182) < 1E-4:  # Including from 2.7182
            return True, 'C_e'

        # If the value is not an integer, we write it and trim followed zeros.
        # We need to use '%.15f' formatting because str() may gives string using precisions like 1.7E-3
        # Also we will trim after four zeros under the decimal like 0.05000000074 because of float's precision.
        return const >= 0, 'C_%s' % \
               EPT.FOLLOWING_ZERO_PATTERN.sub('\\1', ('%.15f' % abs(const)).replace('.', '_'))

def orig_infix_to_postfix(equation: Union[str, List[str]], number_token_map: dict, free_symbols: list,
                     join_output: bool = True):
    """
    Read infix equation string and convert it into a postfix string

    :param Union[str,List[str]] equation:
        Either one of these.
        - A single string of infix equation. e.g. "5 + 4"
        - Tokenized sequence of infix equation. e.g. ["5", "+", "4"]
    :param dict number_token_map:
        Mapping from a number token to its anonymized representation (e.g. N_0)
    :param list free_symbols:
        List of free symbols (for return)
    :param bool join_output:
        True if the output need to be joined. Otherwise, this method will return the tokenized postfix sequence.
    :rtype: Union[str, List[str]]
    :return:
        Either one of these.
        - A single string of postfix equation. e.g. "5 4 +"
        - Tokenized sequence of postfix equation. e.g. ["5", "4", "+"]
    """
    # Template in ALG514/DRAW is already tokenized, without parenthesis.
    # Template in MAWPS is also tokenized but contains parenthesis.
    output_tokens = []
    postfix_stack = []

    # Tokenize the string if that is not tokenized yet.
    if type(equation) is str:
        equation = equation.strip().split(' ')

    # Read each token
    for tok in equation:
        # Ignore blank token
        if not tok:
            continue

        if tok == ')':
            # Pop until find the opening paren '('
            while postfix_stack:
                top = postfix_stack.pop()
                if top == '(':
                    # Discard the matching '('
                    break
                else:
                    output_tokens.append(top)
        elif tok in '*/+-=(':
            # '(' has the highest precedence when in the input string.
            precedence = EPT.OPERATOR_PRECEDENCE.get(tok, 1E9)

            while postfix_stack:
                # Pop until the top < current_precedence.
                # '(' has the lowest precedence in the stack.
                top = postfix_stack[-1]
                if EPT.OPERATOR_PRECEDENCE.get(top, -1E9) < precedence:
                    break
                else:
                    output_tokens.append(postfix_stack.pop())
            postfix_stack.append(tok)
        elif EPT.NUMBER_PATTERN.fullmatch(tok) is not None:
            # Just output the operand.
            positive, const_code = constant_number(tok)
            if not positive:
                const_code = const_code + '_NEG'
            output_tokens.append(const_code)
        elif tok in number_token_map:
            # Just output the operand
            output_tokens += number_token_map[tok]
        else:
            # This is a variable name
            # Just output the operand.
            if tok not in free_symbols:
                free_symbols.append(tok)

            tok = 'X_%s' % free_symbols.index(tok)
            output_tokens.append(tok)

    while postfix_stack:
        output_tokens.append(postfix_stack.pop())

    if join_output:
        return ' '.join(output_tokens)
    else:
        return output_tokens

def refine_formula_as_prefix(item, numbers, dataset_name):
    formula = item['infix equation']
    template_to_number = {}
    template_to_value = {}
    
    number_by_tokenid = {j: i for i, x in enumerate(numbers) for j in x['token']}

    for tokid, token in enumerate(re.sub('\\s+', ' ', item['aux']['mask_text']).strip().split(' ')):
        if token.startswith('temp_'):
            assert tokid in number_by_tokenid, (tokid, number_by_tokenid, item['aux'])

            num_id = number_by_tokenid[tokid]
            template_to_number[token] = ['N_%s' % num_id]
            template_to_value[token] = numbers[num_id]['value']

    # We should read both template_equ and new_equation because of NONE in norm_post_equ.
    formula = item['aux']['template_equ'].split(' ')
    original = item['aux']['new_equation'].split(' ')
    assert len(formula) == len(original)

    # Recover 'NONE' constant in the template_equ.
    for i in range(len(formula)):
        f_i = formula[i]
        o_i = original[i]

        if f_i == 'NONE':
            formula[i] = original[i]
        elif f_i.startswith('temp_'):
            assert abs(float(template_to_value[f_i]) - float(o_i)) < 1E-4,\
                "Equation is different! '%s' vs '%s' at %i-th position" % (formula, original, i)
        else:
            # Check whether two things are the same.
            assert f_i == o_i, "Equation is different! '%s' vs '%s' at %i-th position" % (formula, original, i)

    free_symbols = []
    new_formula = [(EPT.PREP_KEY_EQN, orig_infix_to_postfix(formula, template_to_number, free_symbols))]

    if free_symbols:
        new_formula.append((EPT.PREP_KEY_ANS, ' '.join(['X_%s' % i for i in range(len(free_symbols))])))

    return new_formula

def ept_preprocess(datas, dataset_name):
    datas_list = []
    
    
    for idx, data in enumerate(datas):
        answer_list = [(x,) for x in data['aux']['lSolutions']]
        masked_text = re.sub('\\s+', ' ', data['aux']['mask_text']).strip().split(' ')
        temp_tokens = data['aux']['num_list']

        regenerated_text = []
        for token in masked_text:
            if token.startswith('temp_'):
                regenerated_text.append(str(temp_tokens[int(token[5:])]))
            else:
                regenerated_text.append(token)

        problem = ' '.join(regenerated_text)

        text, numbers = find_ept_numbers_in_text(problem)
        data['ept'] = {}
        data['ept']['text'] = text

        data['ept']['numbers'] = numbers
        
        data['ept']['answer'] = answer_list
        prefix_formula = refine_formula_as_prefix(data, numbers, dataset_name)
        data['ept']['expr'] = prefix_formula
        
        datas_list.append(data)
    return datas_list

def preprocess_ept_dataset_(train_datas, valid_datas, test_datas, dataset_name):
    train_datas = ept_preprocess(train_datas, dataset_name)
    valid_datas = ept_preprocess(valid_datas, dataset_name)
    test_datas = ept_preprocess(test_datas, dataset_name)
    return train_datas, valid_datas, test_datas

def ept_equ_preprocess(formulae, decoder):
    if decoder == 'vall':
        assert type(formulae) is list, "We expect [(TYPE, EQUATION), ...] " \
                                       "where TYPE = 0, 1, 2 and EQUATION is a list of tokens."

        tokens = []
        memory_counter = 0
        variables = {}

        for typ, expr in formulae:
            if type(expr) is str:
                expr = re.split('\\s+', expr.strip())

            if typ == EPT.PREP_KEY_ANS:
                # Ignore answer tuple
                continue
            elif typ == EPT.PREP_KEY_MEM:
                # If this is a memory, then make it as M_<id> = <expr>.
                expr = ['M_%s' % memory_counter] + expr + ['=']
                memory_counter += 1

            for token in expr:
                # Normalize tokens
                if any(token.startswith(prefix) for prefix in ['X_']):
                    # Remapping variables by order of appearance.
                    if token not in variables:
                        variables[token] = len(variables)

                    position = variables[token]
                    token = EPT.FORMAT_VAR % position  # By the index of the first appearance.
                    tokens.append(token)
                elif any(token.startswith(prefix) for prefix in ['NUM_']):
                    # To preserve order, we padded indices with zeros at the front.
                    position = int(token.split('_')[-1])
                    tokens.append('NUM_%d' % position)
                else:
                    if token.startswith('C_'):
                        token = token.replace('C_', EPT.CON_PREFIX)
                    tokens.append(token)
        return tokens
    elif decoder == 'expr_gen':
        assert type(formulae) is list, "We expect [(TYPE, EQUATION), ...] " \
                                       "where TYPE = 0, 1, 2 and EQUATION is a list of tokens."

        variables = []
        memories = []

        for typ, expr in formulae:
            if type(expr) is str:
                expr = re.split('\\s+', expr.strip())

            # Replace number, const, variable tokens with N_<id>, C_<value>, X_<id>
            normalized = []
            for token in expr:
                if any(token.startswith(prefix) for prefix in ['X_']):
                    # Case 1: Variable
                    if token not in variables:
                        variables.append(token)

                    # Set as negative numbers, since we don't know how many variables are in the list.
                    normalized.append((EPT.ARG_MEM, - variables.index(token) - 1))
                elif any(token.startswith(prefix) for prefix in ['N_']):
                    # Case 2: Number
                    token = int(token.split('_')[-1])
                    normalized.append((EPT.ARG_NUM, EPT.FORMAT_NUM % token))

                elif token.startswith('C_'):
                    normalized.append((EPT.ARG_CON, token.replace('C_', EPT.CON_PREFIX)))
                else:
                    normalized.append(token)

            # Build expressions (ignore answer tuples)
            if typ == EPT.PREP_KEY_EQN:
                stack_len = postfix_parser(normalized, memories)
                assert stack_len == 1, "Equation is not correct! '%s'" % expr
            elif typ == EPT.PREP_KEY_MEM:
                stack_len = postfix_parser(normalized, memories)
                assert stack_len == 1, "Intermediate representation of memory is not correct! '%s'" % expr

        # Reconstruct indices for result of prior expression.
        var_length = len(variables)
        # Add __NEW_VAR at the front of the sequence. The number of __NEW_VAR()s equals to the number of variables used.
        preprocessed = [(EPT.FUN_NEW_VAR, []) for _ in range(var_length)]
        for operator, operands in memories:
            # For each expression
            new_arguments = []
            for typ, tok in operands:
                if typ == EPT.ARG_MEM:
                    # Shift index of prior expression by the number of variables.
                    tok = tok + var_length if tok >= 0 else -(tok + 1)

                    tok = EPT.FORMAT_MEM % tok

                new_arguments.append((typ, tok))

            # Register an expression
            preprocessed.append((operator, new_arguments))

        return preprocessed
    else:
        assert type(formulae) is list, "We expect [(TYPE, EQUATION), ...] " \
                                       "where TYPE = 0, 1, 2 and EQUATION is a list of tokens."

        variables = []
        memories = []

        for typ, expr in formulae:
            if type(expr) is str:
                expr = re.split('\\s+', expr.strip())

            # Replace number, const, variable tokens with N_<id>, C_<value>, X_<id>
            normalized = []
            for token in expr:
                if any(token.startswith(prefix) for prefix in ['X_']):
                    # Case 1: Variable
                    if token not in variables:
                        variables.append(token)

                    # Set as negative numbers, since we don't know how many variables are in the list.
                    normalized.append((EPT.ARG_MEM, - variables.index(token) - 1))
                elif any(token.startswith(prefix) for prefix in ['N_']):
                    # Case 2: Number
                    token = int(token.split('_')[-1])
                    normalized.append((EPT.ARG_NUM, token))

                elif token.startswith('C_'):
                    normalized.append((EPT.ARG_CON, token.replace('C_', EPT.CON_PREFIX)))
                else:
                    normalized.append(token)

            # Build expressions (ignore answer tuples)
            if typ == EPT.PREP_KEY_EQN:
                stack_len = postfix_parser(normalized, memories)
                assert stack_len == 1, "Equation is not correct! '%s'" % expr
            elif typ == EPT.PREP_KEY_MEM:
                stack_len = postfix_parser(normalized, memories)
                assert stack_len == 1, "Intermediate representation of memory is not correct! '%s'" % expr

        # Reconstruct indices for result of prior expression.
        var_length = len(variables)
        # Add __NEW_VAR at the front of the sequence. The number of __NEW_VAR()s equals to the number of variables used.
        preprocessed = [(EPT.FUN_NEW_VAR, []) for _ in range(var_length)]
        for operator, operands in memories:
            # For each expression
            new_arguments = []
            for typ, tok in operands:
                if typ == EPT.ARG_MEM:
                    # Shift index of prior expression by the number of variables.
                    tok = tok + var_length if tok >= 0 else -(tok + 1)

                new_arguments.append((typ, tok))

            # Register an expression
            preprocessed.append((operator, new_arguments))

        return preprocessed

def pad_token_ept_inp(ques_batch, tokenizer, num_list_batch):
    max_len = max(len(x) - x.count(EPT.NUM_TOKEN) for x in ques_batch)

    max_len = min(max_len, 510)

    # Maximum sequence length with BOS and EOS
    max_len_with_specials = max_len + 2
    # Storage for padded values
    padded = []
    numbers = []
    num_pos = []

    # Shortcut for BOS, EOS, PAD token
    bos_token = "[CLS]"
    eos_token = "[SEP]"
    pad_token = "<pad>"

    for item_id, item in enumerate(ques_batch):
        tokens = []
        number_indicators = []
        number_index = 0
        # We add tokens except [NUM], which we added to mark the position of numbers
        item = tokenizer.convert_ids_to_tokens(item)
        for tok in item:
            if tok != EPT.NUM_TOKEN:
                # If this is not a [NUM] token, just add it.
                tokens.append(tok)
                # We don't know whether the token is representing a number or not yet, so set it as PAD
                number_indicators.append(EPT.PAD_ID)
            else:
                # If this is a [NUM] token, then previous tokens that form a single word are representing numbers.
                # Set number index until we meet SPIECE_UNDERLINE (Beginning of a word).
                for i in range(-1, -len(tokens) - 1, -1):
                    # From -1 to -len(tok) (check a token backward)
                    if tokens[i] != EPT.SPIECE_UNDERLINE:
                        # We ignore SPIECE_UNDERLINE token when marking the position of numbers.
                        # Note that this code does not ignore tokens starting with SPIECE_UNDERLINE.
                        number_indicators[i] = number_index

                    if tokens[i].startswith(EPT.SPIECE_UNDERLINE):
                        # Break when we meet the beginning of a word.
                        break

                # Increase index of written numbers
                number_index += 1

        # Check whether any number token is discarded.
        assert max(number_indicators[max_len:], default=EPT.PAD_ID) == EPT.PAD_ID, \
            "A number token should not be discarded. You should increase the number of input tokens."

        assert number_index == len(num_list_batch[item_id]) and len(set(number_indicators)) - 1 == number_index, \
            "The extracted numbers are not the same! %s vs %s" % (number_index, len(num_list_batch[item_id]))

        # Build tokens
        tokens = [bos_token] + tokens[:max_len] + [eos_token]
        number_indicators = [EPT.PAD_ID] + number_indicators[:max_len] + [EPT.PAD_ID]

        # Pad and append the item
        remain_len = max(0, max_len_with_specials - len(tokens))
        padded.append(tokens + [pad_token] * remain_len)
        num_pos.append(number_indicators + [EPT.PAD_ID] * remain_len)


    return padded, num_pos

def read_aux_jsonl_data(aux_dataset_file):
    _dataset = []
    with Path(aux_dataset_file).open('r+t', encoding='UTF-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            _dataset.append(item['original'])
    return _dataset
'''
{
    "original": {
        "sQuestion": "7 audio cassettes and 3 video cassettes cost rs 1110 , while 5 audio cassettes and 4 video cassettes cost rs 1350 . Find the cost of an audio cassette and a video cassette .",
        "lSolutions": [30.0, 300.0], 
        "Template": ["a * m + b * n = c", "d * m + e * n = f"],
        "lEquations": ["(7.0*audio_cassettes)+(3.0*video_cassettes)=1110.0", "(5.0*audio_cassettes)+(4.0*video_cassettes)=1350.0"], 
        "iIndex": 5484, 
        "Alignment": [
            {"coeff": "a", "SentenceId": 0, "Value": 7.0, "TokenId": 0}, 
            {"coeff": "b", "SentenceId": 0, "Value": 3.0, "TokenId": 4}, 
            {"coeff": "c", "SentenceId": 0, "Value": 1110.0, "TokenId": 9}, 
            {"coeff": "d", "SentenceId": 0, "Value": 5.0, "TokenId": 12}, 
            {"coeff": "e", "SentenceId": 0, "Value": 4.0, "TokenId": 16}, 
            {"coeff": "f", "SentenceId": 0, "Value": 1350.0, "TokenId": 21}
            ], 
        "Equiv": []
    }, 
    "text": "7 audio cassettes and 3 video cassettes cost rs 1110 , while 5 audio cassettes and 4 video cassettes cost rs 1350 . Find the cost of an audio cassette and a video cassette .", 
    "numbers": [{"token": [0], "value": "7", "is_text": false, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}, {"token": [4], "value": "3", "is_text": false, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}, {"token": [9], "value": "1110", "is_text": false, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}, {"token": [12], "value": "5", "is_text": false, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}, {"token": [16], "value": "4", "is_text": false, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}, {"token": [21], "value": "1350", "is_text": false, "is_integer": true, "is_ordinal": false, "is_fraction": false, "is_single_multiple": false, "is_combined_multiple": false}], 
    "answer": [[30.0, 300.0]], 
    "expr": [[0, "N_0 X_0 * N_1 X_1 * + N_2 ="], [0, "N_3 X_0 * N_4 X_1 * + N_5 ="], [1, "X_0 X_1"]], 
    "id": "alg514_fold3_train-05484", 
    "set": "alg514_fold3_train"
}
'''