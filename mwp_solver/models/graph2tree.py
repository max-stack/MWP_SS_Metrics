import copy
import itertools
import torch
import stanza
from torch import nn
from torch.nn import functional

from module.Encoder.graph_based_encoder import GraphBasedEncoder
from module.Embedder.basic_embedder import BasicEmbedder
from module.Decoder.tree_decoder import TreeDecoder
from module.Layer.tree_layers import TreeNode, TreeEmbedding, Prediction, GenerateNode, Merge
from module.Strategy.beam_search import TreeBeam
from loss.masked_cross_entropy_loss import MaskedCrossEntropyLoss, masked_cross_entropy
from utils.enum_type import SpecialTokens, NumMask
from utils.utils import str2float, copy_list

class Graph2Tree(nn.Module):
    def __init__(self, config, dataset):
        super(Graph2Tree, self).__init__()
        self.device = config["device"]
        self.USE_CUDA = True if self.device == torch.device("cuda") else False
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.dropout_ratio = config["dropout_ratio"]
        self.num_layers = config["num_layers"]
        self.rnn_cell_type = config["rnn_cell_type"]
        self.bidirectional = config["bidirectional"]
        self.language = config["language"]
        self.beam_size = config["beam_size"]
        self.max_output_len = config["max_output_len"]
        self.embedding = config["embedding"]

        self.vocab_size = len(dataset.in_idx2word)
        self.num_start = dataset.num_start

        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)
        self.mask_list = NumMask.number

        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        #module
        
        self.embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.encoder=GraphBasedEncoder(self.embedding_size,self.hidden_size,self.rnn_cell_type,\
                                        self.bidirectional,self.num_layers,self.dropout_ratio)
        self.decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)

        self.loss = MaskedCrossEntropyLoss()

    def calculate_loss(self, batch_data:dict) -> float:
        """Finish forward-propagating, calculating loss and back-propagation.
        :param batch_data: one batch data.
        :return: loss value.
        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos', 'num list', 'group nums'
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        target_length = torch.LongTensor(batch_data["equ len"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_size = batch_data["num size"]
        num_pos = batch_data["num pos"]
        num_list = batch_data['num list']
        group_nums = batch_data['group nums']

        generate_nums = self.generate_nums
        num_start = self.num_start
        unk = self.unk_token
        graph = self.build_graph(seq_length, num_list, num_pos, group_nums)

        loss = self.train_tree(seq,seq_length,target,target_length,\
            nums_stack,num_size,graph,generate_nums,num_pos,unk,num_start)
        return loss

    def model_test(self, batch_data:dict) -> tuple:
        """Model test.
        
        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list', 'group nums'
        """
        seq = torch.tensor(batch_data["question"]).to(self.device)
        seq_length = torch.tensor(batch_data["ques len"]).long()
        target = torch.tensor(batch_data["equation"]).to(self.device)
        nums_stack = copy.deepcopy(batch_data["num stack"])
        num_pos = batch_data["num pos"]
        target = batch_data["equation"]
        num_list = batch_data['num list']
        group_nums = batch_data['group nums']

        generate_nums = self.generate_nums
        num_start = self.num_start
        graph = self.build_graph(seq_length, num_list, num_pos, group_nums)
        all_node_output = self.evaluate_tree(seq, seq_length, graph, generate_nums, num_pos, num_start, self.beam_size, self.max_output_len)

        all_output = self.convert_idx2symbol(all_node_output, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def train_tree(self, input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, graph, generate_nums, num_pos, unk, num_start, english=False):
        # sequence mask for attention
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask)

        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_nums)
        for i in num_size_batch:
            d = i + len(generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)

        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = input_batch.transpose(0, 1)

        target = target_batch.transpose(0, 1)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)
        batch_size = len(input_length)

        if self.USE_CUDA:
            input_var = input_var.cuda()
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()

        # Run words through encoder
        seq_emb = self.embedder(input_var)
        encoder_outputs, problem_output = self.encoder(seq_emb, input_length, graph)

        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, self.hidden_size)

        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask,
                                                                                                       num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = self.generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
            target[t] = target_t
            if self.USE_CUDA:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous()
        if self.USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
            # target_length = torch.LongTensor(target_length).cuda()
        else:
            pass
            # target_length = torch.LongTensor(target_length)
        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        loss = masked_cross_entropy(all_node_outputs, target, target_length)
        # loss = loss_0 + loss_1
        loss.backward()
        # clip the grad
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

        return loss.item()  # , loss_0.item(), loss_1.item()

    def evaluate_tree(self, input_batch, input_length, graph, generate_nums, num_pos, num_start, beam_size=5, max_length=30):
        seq_mask = torch.BoolTensor(1, input_length).fill_(0)
        # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
        input_var = input_batch.transpose(0, 1)

        num_mask = torch.BoolTensor(1, len(num_pos[0]) + len(generate_nums)).fill_(0)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)

        batch_size = 1

        if self.USE_CUDA:
            input_var = input_var.cuda()
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()
        # Run words through encoder

        seq_emb = self.embedder(input_var)
        encoder_outputs, problem_output = self.encoder(seq_emb, input_length, graph)

        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        num_size = len(num_pos[0])
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, self.hidden_size)
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                                                                                                           seq_mask, num_mask)

                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if self.USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0].out

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [1 for _ in range(hidden_size)]
        temp_0 = [0 for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [0 for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices.cuda()
            masked_index = masked_index.cuda()
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        # when the decoder input is copied num but the num has two pos, chose the max
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float("1e12")
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def build_graph(self, seq_length, num_list, num_pos, group_nums):
        max_len = seq_length.max()
        batch_size = len(seq_length)
        batch_graph = []
        for b_i in range(batch_size):
            x = torch.zeros((max_len, max_len))
            for idx in range(seq_length[b_i]):
                x[idx, idx] = 1
            quantity_cell_graph = torch.clone(x)
            graph_greater = torch.clone(x)
            graph_lower = torch.clone(x)
            graph_quanbet = torch.clone(x)
            graph_attbet = torch.clone(x)
            for idx, n_pos in enumerate(num_pos[b_i]):
                for pos in group_nums[b_i][idx]:
                    quantity_cell_graph[n_pos, pos] = 1
                    quantity_cell_graph[pos, n_pos] = 1
                    graph_quanbet[n_pos, pos] = 1
                    graph_quanbet[pos, n_pos] = 1
                    graph_attbet[n_pos, pos] = 1
                    graph_attbet[pos, n_pos] = 1
            for idx_i in range(len(num_pos[b_i])):
                for idx_j in range(len(num_pos[b_i])):
                    num_i = str2float(num_list[b_i][idx_i])
                    num_j = str2float(num_list[b_i][idx_j])
                    if num_i > num_j:
                        graph_greater[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
                        graph_lower[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                    else:
                        graph_greater[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                        graph_lower[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
            group_num_ = itertools.chain.from_iterable(group_nums[b_i])
            combn = itertools.permutations(group_num_, 2)
            for idx in combn:
                graph_quanbet[idx] = 1
                graph_quanbet[idx] = 1
                graph_attbet[idx] = 1
                graph_attbet[idx] = 1
            quantity_cell_graph = quantity_cell_graph.to(self.device)
            graph_greater = graph_greater.to(self.device)
            graph_lower = graph_lower.to(self.device)
            graph_quanbet = graph_quanbet.to(self.device)
            graph_attbet = graph_attbet.to(self.device)
            graph = torch.stack([quantity_cell_graph, graph_greater, graph_lower, graph_quanbet, graph_attbet], dim=0)
            batch_graph.append(graph)
        batch_graph = torch.stack(batch_graph)
        return batch_graph

    def convert_idx2symbol(self, output, num_list, num_stack):
        #batch_size=output.size(0)
        '''batch_size=1'''
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if "NUM" in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list
