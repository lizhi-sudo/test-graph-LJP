import math
from itertools import permutations
from copy import deepcopy
import numpy as np
from numpy import average
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor

class BertMatchingGraph(nn.Module):
    def __init__(self,
                 device,
                 num_classes_accu=None,
                 num_classes_law=None,
                 num_classes_term=None,
                 args=None):
        super(BertMatchingGraph, self).__init__()
        self.model = AutoModel.from_pretrained('./pre_model/bert-base-chinese')
        # self.model = AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        
        self.device = device
        self.num_classes_accu = num_classes_accu
        self.num_classes_law = num_classes_law
        self.num_classes_term = num_classes_term
        self.args = args
        self.num_classes = num_classes_law + num_classes_accu + num_classes_term

        # Label Embedding
        self.law_embedding = nn.Parameter(torch.zeros(num_classes_law, 768))
        nn.init.kaiming_uniform_(self.law_embedding, mode='fan_in')
        self.accu_embedding = nn.Parameter(torch.zeros(num_classes_accu, 768))
        nn.init.kaiming_uniform_(self.accu_embedding, mode='fan_in')
        self.term_embedding = nn.Parameter(torch.zeros(num_classes_term, 768))
        nn.init.kaiming_uniform_(self.term_embedding, mode='fan_in')

        self.segment_embedding = nn.Embedding(3, 768)

        # Different sources information alignment
        self.case_embedding_transform = nn.Linear(768, 768)

        # DAG mask
        # positions with ``True`` is not allowed to attend in Transformer block
        # while ``False`` values will be unchanged.
        if args.graph == 'no_edge':
            self.graph_mask = self.get_no_edge_mask()
        elif args.graph == 'LCT':
            self.graph_mask = self.get_dag_mask()
        elif args.graph == 'LCT-CL':
            self.graph_mask = self.get_dag_mask(dag_type='CL')
        elif args.graph == 'LCT-LT':
            self.graph_mask = self.get_dag_mask(dag_type='LT')
        elif args.graph == 'complete_graph':
            self.graph_mask = self.get_complete_graph_mask()
        elif args.graph == 'LCTCL':
            self.graph_mask = self.get_udg_mask()
        elif args.graph == 'LCT-CL-LT':
            self.graph_mask = self.get_dag_mask(dag_type='CL-LT')
        elif args.graph == 'LCTLTCL':
            self.graph_mask = self.get_udg_mask(udg_type='LT')
        elif args.graph == 'dag_intra':
            self.graph_mask = self.get_dag_intra_mask()
        elif args.graph == 'udg_intra':
            self.graph_mask = self.get_udg_intra_mask()
        else:
            raise NameError
        print(f'Graph Mask: {args.graph}')
        if args.graph != 'complete_graph':
            assert (~self.graph_mask
                    ).sum() < self.graph_mask.sum(), self.graph_mask.sum()
            assert self.graph_mask.dtype == torch.bool, self.graph_mask.dtype

            import seaborn as sns
            ax = sns.heatmap(np.array(self.graph_mask))
            file_name = f'{args.graph}_big' if 'big' in args.train_path else f'{args.graph}_small'
            ax.get_figure().savefig(f'./visualize_result/label_relation/{file_name}.png', bbox_inches='tight', dpi = 150)

        features_dim = 768
        if self.args.difference_module:
            graph_encoder_layer = DifferenceModule(features_dim, args, num_classes_law, num_classes_accu, num_classes_term)
        else:
            graph_encoder_layer = nn.TransformerEncoderLayer(
                d_model=features_dim,
                nhead=16,
                dropout=args.dropout_rate,
                batch_first=True)
            graph_encoder_layer.self_attn.dropout = 0  # The defaulted dropout rate of self-attention module is set to args.dropout_rate and we reset it to zero because we don't want the edges of the relation graph be removed randomly, which is unreasonable.

        self.relation_model = nn.TransformerEncoder(
            graph_encoder_layer,
            num_layers=3)

        # Loss
        self.CE_loss = nn.CrossEntropyLoss()

        # Discriminator
        if args.mi_type == 'v0':
            pass
        elif args.mi_type == 'v1':
            self.mutual_information_loss = self.mutual_information_loss_v1
        elif args.mi_type == 'v2':
            self.mutual_information_loss = self.mutual_information_loss_v2
        elif args.mi_type == 'InfoNCE':
            self.mutual_information_loss = self.InfoNCE_loss
        elif args.mi_type == 'sup_contrast':
            if args.proj:
                self.proj = nn.Linear(768, 128)
            self.mutual_information_loss = self.sup_contrast
        elif args.mi_type == 'sup_contrast_local':
            if args.proj:
                self.proj = nn.Linear(768, 128)
            self.mutual_information_loss = self.sup_contrast_local
        elif args.mi_type == 'sup_contrast_global':
            if args.proj:
                self.proj = nn.Linear(768, 128)
            self.mutual_information_loss = self.sup_contrast_global
        elif args.mi_type == 'sup_contrast_task_global':
            if args.proj:
                self.proj = nn.Linear(768, 128)
            self.mutual_information_loss = self.sup_contrast_task_global
        elif args.mi_type == 'traditional_sup_contrast':
            if args.proj:
                self.proj = nn.Linear(768, 128)
            self.mutual_information_loss = self.traditional_sup_contrast
        elif args.mi_type == 'graph_mae':
            # GraphMAE
            graph_encoder_layer = nn.TransformerEncoderLayer(
                d_model=features_dim,
                nhead=16,
                dropout=0,
                batch_first=True)
            self.mae_decoder = nn.TransformerEncoder(
                graph_encoder_layer, num_layers=1)
            self.mask_code = nn.Parameter(torch.zeros(features_dim))
        print(f'Mutual Information Type: {args.mi_type}')

        self.law_matching = nn.Linear(in_features=features_dim,
                                out_features=1)
        # self.accu_matching = self.law_matching
        # self.term_matching = self.law_matching
        self.accu_matching = nn.Linear(in_features=features_dim,
                                out_features=1)
        self.term_matching = nn.Linear(in_features=features_dim,
                                out_features=1)

        # Froze bert to fasten the training process.
        if args.froze_bert:
            self.trainable_param_names = [
                'layer.11', 'layer.10', 'layer.9', 'layer.8', 'layer.7', 'layer.6'
            ]
            for name, param in self.model.named_parameters():
                if any(n in name for n in self.trainable_param_names):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self,
                facts,
                tokenizer,
                labels_accu=None,
                labels_law=None,
                labels_term=None):
        # move data to device
        B = len(facts)

        # tokenize the data text
        inputs = tokenizer(list(facts),
                           max_length=self.args.input_max_length,
                           padding=True,
                           truncation=True,
                           return_tensors='pt')

        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        if self.training and self.args.mi_type in [
                'traditional_sup_contrast', 'sup_contrast_local', 'sup_contrast_global', 'sup_contrast_task_global'
        ]:
            '''
            The dropout in the relation model will produce randomness to the representation, therefore can achieve inherent augmentation for the same sample after different forward
        process. 
            '''
            case_embeddings1 = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)['pooler_output']
            
            case_embeddings2 = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)['pooler_output']

            case_embeddings = torch.cat([case_embeddings1, case_embeddings2], dim=0)
            node_embeddings = self.node_embedding(case_embeddings)
                
            relation_output = self.GNN_forward(node_embeddings)

            # assert not all((case_embeddings1==case_embeddings2).detach().cpu().view(-1).numpy().tolist()), 'Bug: case_embeddings1 == case_embeddings2'
            # assert not all((relation_output1==relation_output2).detach().cpu().view(-1).numpy().tolist()), 'Bug: relation_output1 == relation_output2'

            labels_law = torch.cat([labels_law, labels_law], dim=0)
            labels_accu = torch.cat([labels_accu, labels_accu], dim=0)
            labels_term = torch.cat([labels_term, labels_term], dim=0)

        else:
            case_embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)['pooler_output']
            node_embeddings = self.node_embedding(case_embeddings)
            relation_output = self.GNN_forward(node_embeddings)

        output_law, output_accu, output_term = self.unpacked_according_to_three_types_of_labels(
            relation_output)

        logits_law = self.law_matching(output_law).squeeze(2)
        logits_accu = self.accu_matching(output_accu).squeeze(2)
        logits_term = self.term_matching(output_term).squeeze(2)

        # import pickle
        # data = {'facts': facts,
        # 'labels_accu': labels_accu.cpu().numpy(),
        # 'labels_law': labels_law.cpu().numpy(),
        # 'labels_term': labels_term.cpu().numpy(),
        # 'output_law': output_law.detach().cpu().numpy(),
        # 'output_accu': output_accu.detach().cpu().numpy(),
        # 'output_term': output_term.detach().cpu().numpy(),
        # 'logits_law': logits_law.detach().cpu().numpy(),
        # 'logits_accu': logits_accu.detach().cpu().numpy(),
        #  'logits_term': logits_term.detach().cpu().numpy()}

        # save_file = './visualize_result/forward_ckpt_udg_global.pkl'
        # with open(save_file, 'wb') as f:
        #     pickle.dump(data, f)

        # exit(-1)

        if self.training and labels_law is not None:
            alpha = 1
            loss_law = self.CE_loss(logits_law, labels_law)
            loss_accu = self.CE_loss(logits_accu, labels_accu)
            loss_term = self.CE_loss(logits_term, labels_term)
            # TODO: multiply loss_term by 2 to fasten the optimization?
            loss = loss_law + loss_accu + alpha * loss_term                    

            if self.args.mi_type == 'v0':
                return loss, logits_law, logits_accu, logits_term
            elif self.args.mi_type in [
                    'traditional_sup_contrast', 'sup_contrast_local',
                    'sup_contrast_global', 'sup_contrast_task_global'
            ]:
                loss_sup_contrast = self.mutual_information_loss(
                    relation_output,
                    labels_law=labels_law,
                    labels_accu=labels_accu,
                    labels_term=labels_term)

                return loss, loss_sup_contrast, logits_law[:
                                                           B], logits_accu[:
                                                                           B], logits_term[:
                                                                                           B]
            elif self.args.mi_type == 'graph_mae':
                loss_mae = self.graph_mae_forward_and_loss(node_embeddings, relation_output, self.args.mask_ratio)
                return loss, loss_mae, logits_law, logits_accu, logits_term

        return logits_law, logits_accu, logits_term

    def node_embedding(self, case_embeddings):
        # Input
        node_embedding = torch.cat(
            [self.law_embedding, self.accu_embedding, self.term_embedding])
        segment_id = torch.cat([
            torch.ones(self.num_classes_law).long() * 0,
            torch.ones(self.num_classes_accu).long() * 1,
            torch.ones(self.num_classes_term).long() * 2
        ]).to(self.device)
        segment_embedding = self.segment_embedding(segment_id).unsqueeze(0)

        # case_embedding = torch.mean(pooler_output, dim=1, keepdim=True)  # [B, 1, 768] TODO: max?
        # TODO: Sentence Whitening

        input_embedding = self.case_embedding_transform(
            case_embeddings).unsqueeze(1)
        node_embedding = input_embedding + segment_embedding + node_embedding.unsqueeze(0)  # [B, L, 768]
        return node_embedding

    def GNN_forward(self, node_embedding):
        """
        args:
        node_embedding: [B, L, C]

        """
        # attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
        # source sequence length.
        # if self.args.GNN:
        #     relation_output = self.relation_model(input_embedding)
        # else:
        relation_output = self.relation_model(
            node_embedding,
            mask=self.graph_mask.transpose(0, 1).to(self.device)
            if self.graph_mask is not None else None)

        return relation_output

    def graph_mae_forward_and_loss(self, x, h, mask_ratio=0.1, gamma=1):
        B, N, D = h.shape

        mask, ids_mask = self.random_masking(h, mask_ratio)

        h_mask = h * mask.unsqueeze(-1).repeat(1, 1, D)

        mask_code = self.mask_code * (1 - mask.unsqueeze(-1).repeat(1, 1, D))

        h_mask += mask_code

        z = self.mae_decoder(h_mask, mask=self.graph_mask.transpose(0, 1).to(self.device)
            if self.graph_mask is not None else None)

        # Scaled Cosine Error
        x_masked = torch.gather(x, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, D))
        z_masked = torch.gather(z, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, D))

        cos_similarity = ((x_masked * z_masked).sum(2) / (x_masked.norm(dim=2) * z_masked.norm(dim=2)))
        loss_mae = (mask_ratio * (1 - cos_similarity) ** gamma).sum(1).mean(0)

        return loss_mae
        
    def random_masking(self, x, mask_ratio):
        """

        Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by argsort random noise.

        x: [N, L, D], sequence

        """

        N, L, D = x.shape # batch, length, dim

        len_mask = int(L * mask_ratio)

        noise = torch.rand(N, L) # noise in [0, 1]

        # sort noise for each sample

        ids_shuffle = torch.argsort(

        noise, dim=1) # ascend: small is keep, large is remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # # keep the first subset

        ids_mask = ids_shuffle[:, :len_mask]

        # generate the binary mask: 0 is keep, 1 is remove

        mask = torch.ones([N, L])

        mask[:, :len_mask] = 0

        # unshuffle to get the binary mask

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.to(x.device), ids_mask.to(x.device)

    def unpacked_according_to_three_types_of_labels(self, input):
        return input[:, :self.
                     num_classes_law], input[:, self.num_classes_law:self.
                                             num_classes_law + self.
                                             num_classes_accu], input[:, self.
                                                                      num_classes_law
                                                                      + self.
                                                                      num_classes_accu:
                                                                      self.
                                                                      num_classes_law
                                                                      + self.
                                                                      num_classes_accu
                                                                      + self.
                                                                      num_classes_term]

    def get_complete_graph_mask(self):
        return None

    def get_no_edge_mask(self):
        return (1 - torch.eye(self.num_classes)).bool()

    def get_dag_mask(self, dag_type=None):
        if 'big' in self.args.train_path:
            law_accu_edges = np.loadtxt(
                './datasets/law_accu_edges_big.txt', dtype='int')
            print('Load law_accu_edges_big.txt edges.')
            accu_term_edges = np.loadtxt('./datasets/accu_term_edges_big.txt',
                                         dtype='int')
            print('Load accu_term_edges_big.txt edges.')
        else:
            law_accu_edges = np.loadtxt(
                    './datasets/law_accu_edges.txt',
                    dtype='int')
            print('Load law_accu_edges.txt edges.')
            accu_term_edges = np.loadtxt('./datasets/accu_term_edges.txt',
                                         dtype='int')
            print('Load accu_term_edges.txt edges.')

        if dag_type == 'LT' or dag_type == 'CL-LT':
            if 'big' in self.args.train_path:
                law_term_edges = np.loadtxt(
                    './datasets/law_term_edges_big.txt', dtype='int')
                print('Load law_term_edges_big.txt edges.')
            else:
                law_term_edges = np.loadtxt('./datasets/law_term_edges.txt',
                                            dtype='int')
                print('Load law_term_edges.txt edges.')

        dag_mask = 1 - torch.eye(self.num_classes)
        for i, j in law_accu_edges:
            dag_mask[i, self.num_classes_law + j] = 0
        for i, j in accu_term_edges:
            dag_mask[self.num_classes_law + i,
                     self.num_classes_law + self.num_classes_accu + j] = 0
        if dag_type == 'LT' or dag_type == 'CL-LT':
            for i, j in law_term_edges:
                dag_mask[i,
                         self.num_classes_law + self.num_classes_accu + j] = 0
        if dag_type == 'CL' or dag_type == 'CL-LT':
            for i, j in law_accu_edges:
                dag_mask[self.num_classes_law + j, i] = 0
        return dag_mask.bool()

    def get_udg_mask(self, udg_type=None):
        dag_mask = ~self.get_dag_mask(dag_type=udg_type)
        undirected_relation_mask = dag_mask + dag_mask.transpose(0, 1)
        assert undirected_relation_mask.sum() == (
            dag_mask.sum() + dag_mask.transpose(0, 1).sum() -
            dag_mask.shape[0])

        return ~undirected_relation_mask

    def get_dag_intra_mask(self):
        dag_intra_mask = self.get_dag_mask()
        dag_intra_mask[:self.num_classes_law, :self.num_classes_law] = 0
        dag_intra_mask[self.num_classes_law:self.num_classes_law +
                       self.num_classes_accu,
                       self.num_classes_law:self.num_classes_law +
                       self.num_classes_accu] = 0

        term_begin_idx = self.num_classes_law + self.num_classes_accu
        dag_intra_mask[term_begin_idx:, term_begin_idx:] = 0
        return dag_intra_mask

    def get_udg_intra_mask(self):
        udg_intra_mask = self.get_udg_mask()
        udg_intra_mask[:self.num_classes_law, :self.num_classes_law] = 0
        udg_intra_mask[self.num_classes_law:self.num_classes_law +
                       self.num_classes_accu,
                       self.num_classes_law:self.num_classes_law +
                       self.num_classes_accu] = 0

        term_begin_idx = self.num_classes_law + self.num_classes_accu
        udg_intra_mask[term_begin_idx:, term_begin_idx:] = 0
        return udg_intra_mask

    def traditional_sup_contrast(self,
                     V,
                     labels_law=None,
                     labels_accu=None,
                     labels_term=None,
                     temperature=0.1):
        # V: shape [B, L, C]
        if self.args.proj:
            V = self.proj(V)
        V = F.normalize(V, dim=2)
        B, L, C = V.shape

        # Get positive samples from labels
        labels_accu = labels_accu + self.num_classes_law
        labels_term = labels_term + self.num_classes_law + self.num_classes_accu

        labels = torch.cat([
            labels_law.unsqueeze(1),
            labels_accu.unsqueeze(1),
            labels_term.unsqueeze(1)
        ],
                                           dim=1)  # [B, 3]

        positive_sample_masks = torch.zeros([B, B]).to(self.device)
        for i, sample_label_i in enumerate(labels):
            for j, sample_label_j in enumerate(labels):
                if i != j and (sample_label_i == sample_label_j).all().tolist():
                    positive_sample_masks[i, j] = 1

        V_sample = V.mean(1)  # [B, C]
        relation_score = torch.exp(torch.div(
            torch.matmul(V_sample, V_sample.transpose(0, 1)),
            temperature))  # [B, B]
        positive_score = relation_score * positive_sample_masks
        negative_score = (relation_score * (1 - positive_sample_masks - torch.eye(B).to(self.device))).sum(1, keepdim=True)
        normalized_score = positive_score + negative_score  # B, B

        mean_log_prob = -(torch.log(
            (positive_score + 1e-8) / normalized_score) *
                          positive_sample_masks).sum() / positive_sample_masks.sum()
        return mean_log_prob

    def sup_contrast_local(self,
                           V,
                           labels_law=None,
                           labels_accu=None,
                           labels_term=None,
                           temperature=0.1):
        if self.args.proj:
            V = self.proj(V)
        V = F.normalize(V, dim=2)
        B, L, C = V.shape

        # Get positive samples from labels
        positive_samples = []
        labels_accu = labels_accu + self.num_classes_law
        labels_term = labels_term + self.num_classes_law + self.num_classes_accu

        positive_samples_index = torch.cat([
            labels_law.unsqueeze(1),
            labels_accu.unsqueeze(1),
            labels_term.unsqueeze(1)
        ],
                                           dim=1)  # B, 3

        negative_samples_index = []
        all_index = set(range(self.num_classes))
        for batch_idx in range(B):
            positive_index = set(positive_samples_index[batch_idx].detach().
                                 cpu().numpy().tolist())
            negative_index = all_index - positive_index
            negative_samples_index.append(
                torch.LongTensor(list(negative_index)))
        negative_samples_index = torch.stack(negative_samples_index).to(
            V.device)

        # Get positive representation
        # B, 3, C
        positive_samples = torch.gather(
            V, 1,
            positive_samples_index.unsqueeze(2).expand(B, 3, C))
        # B, self.num_classes-3, C
        negative_samples = torch.gather(
            V, 1,
            negative_samples_index.unsqueeze(2).expand(B, L - 3, C))
        # assert negative_samples.shape[1] == (self.num_classes - 3), negative_samples.shape[1]

        positive_relation = torch.div(
            torch.matmul(positive_samples, positive_samples.transpose(1, 2)),
            temperature)  # B, 3, 3
        nagative_relation = torch.div(
            torch.matmul(positive_samples, negative_samples.transpose(1, 2)),
            temperature)  # B, 3, L-3

        # for numerical stability
        # positive_relation = positive_relation - positive_relation.max(1, keepdim=True).values
        # positive_relation = positive_relation - positive_relation.max(1, keepdim=True).values

        positive_score = torch.exp(positive_relation)
        negative_score = torch.exp(nagative_relation).sum(2, keepdim=True)
        normalized_score = positive_score + negative_score  # B, 3, 3
        positive_mask = (1 -
                         torch.eye(3).to(V.device)).expand_as(positive_score)

        mean_log_prob = -(torch.log(
            (positive_score + 1e-8) / normalized_score) *
                          positive_mask).sum() / positive_mask.sum()
        return mean_log_prob

    def sup_contrast_global(self,
                            V,
                            labels_law=None,
                            labels_accu=None,
                            labels_term=None,
                            temperature=0.1):
        '''
        args:
        V: the output of relation model.
        dag_mask: positions with ``True`` is unchanged, ``False`` means not allowed to attend.
        shape:
        V: [B, L, C] 
        dag_mask: [L, L]
        '''
        if self.args.proj:
            V = self.proj(V)
        V = F.normalize(V, dim=2)
        B, L, C = V.shape

        # Get positive samples from labels
        labels_accu = labels_accu + self.num_classes_law
        labels_term = labels_term + self.num_classes_law + self.num_classes_accu

        positive_samples_index = torch.cat([
            labels_law.unsqueeze(1),
            labels_accu.unsqueeze(1),
            labels_term.unsqueeze(1)
        ],
                                           dim=1)  # B, 3

        negative_samples_index = []
        all_index = set(range(self.num_classes))
        for batch_idx in range(B):
            positive_index = set(positive_samples_index[batch_idx].detach().
                                 cpu().numpy().tolist())
            negative_index = all_index - positive_index
            negative_samples_index.append(
                torch.LongTensor(list(negative_index)))
        negative_samples_index = torch.stack(negative_samples_index).to(
            V.device)

        # Get positive representation
        # B, 3, C
        positive_samples = torch.gather(
            V, 1,
            positive_samples_index.unsqueeze(2).expand(B, 3, C))
        # B, self.num_classes-3, C
        negative_samples = torch.gather(
            V, 1,
            negative_samples_index.unsqueeze(2).expand(B, L - 3, C))
        # assert negative_samples.shape[1] == (self.num_classes - 3), negative_samples.shape[1]

        # Check
        # for batch_idx in range(B):
        #     for i, positive_idx in enumerate(positive_samples_index[batch_idx]):
        #         assert all((positive_samples[batch_idx, i] == V[batch_idx, positive_idx]).detach().cpu().view(-1).numpy().tolist()), "Unmatch"

        positive_samples = positive_samples.view(-1, C)
        negative_samples = negative_samples.view(-1, C)

        positive_relation = torch.div(
            torch.matmul(positive_samples, positive_samples.transpose(0, 1)),
            temperature)  # B*3, B*3
        nagative_relation = torch.div(
            torch.matmul(positive_samples, negative_samples.transpose(0, 1)),
            temperature)  # B*3, B*(L-3)

        # for numerical stability
        # positive_relation = positive_relation - positive_relation.max(1, keepdim=True).values
        # positive_relation = positive_relation - positive_relation.max(1, keepdim=True).values

        positive_score = torch.exp(positive_relation)
        negative_score = torch.exp(nagative_relation).sum(1, keepdim=True)
        normalized_score = positive_score + negative_score  # B*3, B*3
        positive_mask = 1 - torch.eye(B * 3).to(V.device)

        mean_log_prob = -(torch.log(
            (positive_score + 1e-8) / normalized_score) *
                          positive_mask).sum() / positive_mask.sum()
        return mean_log_prob


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, adj):
        super(GraphConvolution, self).__init__()
        self.in_size = in_channels
        self.out_size = out_channels
        self.adj = adj
        self.weight = nn.Parameter(torch.Tensor(self.in_size, self.out_size))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        m = torch.matmul(x, self.weight)
        m = torch.matmul(self.adj, m)
        return m


class GNN(nn.Module):

    def __init__(self, in_channels, hidden_channels, adj):
        super(GNN, self).__init__()
        self.adj_normed = nn.Parameter(self.normalize_adj(adj),
                                       requires_grad=False)
        self.m1 = GraphConvolution(in_channels=in_channels,
                                   out_channels=hidden_channels,
                                   adj=self.adj_normed)
        self.m2 = GraphConvolution(in_channels=hidden_channels,
                                   out_channels=hidden_channels,
                                   adj=self.adj_normed)

    def forward(self, x):
        x = self.m1(x)
        x = F.relu(x, inplace=True)
        x = self.m2(x)
        x = x
        return x

    def normalize_adj(self, A, self_loop=True):
        """Symmetrically normalize adjacency matrix. Used for directed graph.
        Degree_matrix**-0.5 @ Adjacency_matrix @ Degree_matrix**-0.5
        A: adjacency matrix with self loop. shape: [num_in_nodes, num_out_nodes]

        cite: https://www.zhihu.com/question/429974831/answer/1578427205
        """
        if not self_loop:
            A += torch.eye(A.shape[0])
        A = A.transpose(0, 1)  # [num_out_nodes, num_in_nodes]
        D_row_sum = A.sum(1)
        D_inv = 1 / D_row_sum
        D_matrix = torch.diag_embed(D_inv)
        assert D_matrix[1][1] == D_inv[
            1], f'D_inv: {D_inv}, D_matrix: {D_matrix}'
        D_normed = D_matrix @ A
        return D_normed
