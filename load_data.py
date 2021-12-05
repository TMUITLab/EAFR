#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  File name:    load_data.py
  Author:       locke
  Date created: 2020/3/25 下午7:00
"""

import time
import numpy as np
import sklearn
import time
import torch
from sklearn.neighbors import KDTree
import heapq


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

class AlignmentData:

    def __init__(self, data_dir="data/D_W_15K_V1", rate=0.3, share=False, swap=False, val=0.0, with_r=False,OpenEa = False,rev_relation  = True):
        t_ = time.time()

        self.rev_relation = True
        self.rate = rate
        self.val = val
        if(OpenEa):
            self.ins2id_dict, self.id2ins_dict, [self.kg1_ins_ids, self.kg2_ins_ids] = self.OpenEa_load_dict(
                data_dir + "/ent_links")
            self.rel2id_dict, self.id2rel_dict, [self.kg1_rel_ids, self.kg2_rel_ids] = self.OpenEa_load_relation_dict(
                data_dir + "/rel_triples_")
            self.attr2id_dict, self.id2attr_dict, [self.kg1_attr_ids, self.kg2_attr_ids] = self.OpenEa_load_relation_dict(
                data_dir + "/attr_triples_")
            self.ins_num = len(self.ins2id_dict)
            self.rel_num = len(self.rel2id_dict)
            if(self.rev_relation):
                self.rel_num = 2
            self.num_attr = len(self.attr2id_dict)

            self.triple_idx = self.OpenEa_load_triples(data_dir + "/rel_triples_", file_num=2)

            self.ill_idx = self.OpenEa_entities_load_triples(data_dir + "/ent_links", file_num=1)
            self.ill_train_idx = np.array(self.OpenEa_entities_load_triples(data_dir + "/721_5fold/1/train_links", file_num=1))
            #self.ill_val_idx = np.array(self.OpenEa_entities_load_triples(data_dir + "/721_5fold/1/valid_links", file_num=1))
            self.ill_val_idx = []
            self.ill_test_idx = np.array(self.OpenEa_entities_load_triples(data_dir + "/721_5fold/1/test_links", file_num=1))

            self.atrr_idx = self.OpenEa_load_attributes(data_dir + "/attr_triples_", file_num=2)

        else:
            self.ins2id_dict, self.id2ins_dict, [self.kg1_ins_ids, self.kg2_ins_ids] = self.load_dict(data_dir + "/ent_ids_", file_num=2)
            self.rel2id_dict, self.id2rel_dict, [self.kg1_rel_ids, self.kg2_rel_ids] = self.load_dict(data_dir + "/rel_ids_", file_num=2)
            self.ins_num = len(self.ins2id_dict)
            self.rel_num = len(self.rel2id_dict)

            self.triple_idx = self.load_triples(data_dir + "/triples_", file_num=2)
            self.ill_idx = self.load_triples(data_dir + "/ill_ent_ids", file_num=1)

            np.random.shuffle(self.ill_idx)
            self.ill_train_idx, self.ill_val_idx, self.ill_test_idx = np.array(self.ill_idx[:int(len(self.ill_idx) // 1 * rate)], dtype=np.int32), np.array(self.ill_idx[int(len(self.ill_idx) // 1 * rate) : int(len(self.ill_idx) // 1 * (rate+val))], dtype=np.int32), np.array(self.ill_idx[int(len(self.ill_idx) // 1 * (rate+val)):], dtype=np.int32)

        if (self.rev_relation):
            self.rel_num *= 2
            rev_triple_idx = []
            for (h, r, t) in self.triple_idx:
                rev_triple_idx.append((t, r + self.rel_num // 2, h))
            self.triple_idx += rev_triple_idx
        self.ill_idx_dic = {}
        for x in self.ill_idx:
            self.ill_idx_dic[x[0]] = x[1]
            self.ill_idx_dic[x[1]] = x[0]

        self.ins_G_edges_idx, self.ins_G_values_idx, self.r_ij_idx = self.gen_sparse_graph_from_triples(self.triple_idx, self.ins_num, with_r)
        
        assert (share != swap or (share == False and swap == False))

        if share:
            self.triple_idx = self.share(self.triple_idx, self.ill_train_idx)   # 1 -> 2:base
            if(OpenEa):
                self.triple_idx = self.share_attr(self.atrr_idx, self.ill_train_idx)

            self.kg1_ins_ids = (self.kg1_ins_ids - set(self.ill_train_idx[:, 0])) | set(self.ill_train_idx[:, 1])
            self.ill_train_idx = []

        if swap:
            self.triple_idx = self.swap(self.triple_idx, self.ill_train_idx)
            if(OpenEa):
                self.triple_idx = self.swap_attr(self.atrr_idx, self.ill_train_idx)

        self.labeled_alignment = set()
        self.boot_triple_idx = []
        self.boot_pair_dix = []

        self.init_time = time.time() - t_

    def load_triples(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple(map(int, i.split("\t"))) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple

    def load_dict(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        what2id, id2what, ids = {}, {}, []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [i.split("\t") for i in data]
                what2id = {**what2id, **dict([[i[1], int(i[0])] for i in data])}
                id2what = {**id2what, **dict([[int(i[0]), i[1]] for i in data])}
                ids.append(set([int(i[0]) for i in data]))
        return what2id, id2what, ids

    def OpenEa_load_triples(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple([self.ins2id_dict[i.split("\t")[0]],self.rel2id_dict[i.split("\t")[1]],self.ins2id_dict[i.split("\t")[2]]]) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple

    def OpenEa_load_attributes(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple([self.ins2id_dict[i.split("\t")[0]],self.attr2id_dict[i.split("\t")[1]],i.split("\t")[2]]) for i in data]
                triple += data
        triple, _, _ = clear_attribute_triples(triple)
        np.random.shuffle(triple)
        return triple

    def OpenEa_entities_load_triples(self, data_dir, file_num=2):
        if file_num == 2:
            file_names = [data_dir + str(i) for i in range(1, 3)]
        else:
            file_names = [data_dir]
        triple = []
        for file_name in file_names:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip().split("\n")
                data = [tuple([self.ins2id_dict[i.split("\t")[0]],self.ins2id_dict[i.split("\t")[1]]]) for i in data]
                triple += data
        np.random.shuffle(triple)
        return triple

    def OpenEa_load_dict(self, file_name):

        ids = []
        kg1_ents_uri = []
        kg2_ents_uri = []
        ins2id_dict = {}
        id2ins_dict = {}

        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            kg1_ents_uri = [line.split('\t')[0] for line in lines]
            kg2_ents_uri = [line.split('\t')[1] for line in lines]
            id = 0

            for item in kg1_ents_uri:
                if (item not in ins2id_dict):
                    ins2id_dict[item] = id
                    id2ins_dict[id] = item
                    id += 1;
            n1 = id

            for item in kg2_ents_uri:
                if (item not in ins2id_dict):
                    ins2id_dict[item] = id
                    id2ins_dict[id] = item
                    id += 1;
            n2 = id - n1
            ids.append(set([i for i in range(n1)]))
            ids.append(set([i+n1 for i in range(n2)]))

        return ins2id_dict, id2ins_dict, ids

    def OpenEa_load_relation_dict(self, file_name):

        ids = []
        kg1_ents_uri = []
        kg2_ents_uri = []
        ins2id_dict = {}
        id2ins_dict = {}

        id = 0
        pre_n = 0
        for i in range(2):
            with open(file_name + str(i+1), "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
                kg_ents_uri = [line.split('\t')[1] for line in lines]

                n1 = id
                for item in kg_ents_uri:
                    if (item not in ins2id_dict):
                        ins2id_dict[item] = id
                        id2ins_dict[id] = item
                        id += 1;

                ids.append(set([i+n1 for i in range(id-n1)]))

        return ins2id_dict, id2ins_dict, ids




    def recursive_triple_embedding(self,triples, h_embed,r_embed ,t_embed,num_epoch = 2):
        h_embed = sklearn.preprocessing.normalize(h_embed,norm="l2", axis=1)
        t_embed = sklearn.preprocessing.normalize(t_embed, norm="l2", axis=1)
        r_embed = sklearn.preprocessing.normalize(r_embed, norm="l2", axis=1)

        temp_ent_in = h_embed.copy()
        temp_ent_out = t_embed.copy()
        temp_rel_in = h_embed.copy()
        temp_rel_out = t_embed.copy()

        adj_rel_in = np.zeros(h_embed.shape)
        adj_rel_out = np.zeros(h_embed.shape)
        adj_ent_in = np.zeros(h_embed.shape)
        adj_ent_out = np.zeros(h_embed.shape)

        in_deg = np.zeros(h_embed.shape[0])
        out_deg = np.zeros(h_embed.shape[0])
        r_deg = np.zeros(r_embed.shape[0])

        for epoch in range(num_epoch):

            for (h, r, t) in triples:

                adj_rel_in[t]  += (temp_rel_in[r])
                adj_rel_out[h] += (temp_rel_out[r])

                adj_ent_out[h]  += (temp_ent_out[t])
                adj_ent_in[t] += (temp_ent_in[h])
                if(epoch == 0):
                    in_deg[t] += 1
                    out_deg[h] += 1

                    r_deg[r] += 1

            for i in range(h_embed.shape[0]):
                if(out_deg[i] > 0):
                    adj_rel_out[i] +=  adj_rel_out[i] / (out_deg[i] * 2 ** (epoch/2))
                if (in_deg[i] > 0):
                    adj_rel_in[i]  +=  adj_rel_in[i] / (in_deg[i] * 2 ** (epoch/2))
                if(out_deg[i] > 0):
                    adj_ent_out[i] +=  adj_ent_out[i] / (out_deg[i] * 2 ** (epoch/2))
                if (in_deg[i] > 0):
                    adj_ent_in[i]  +=  adj_ent_in[i] / (in_deg[i] * 2 ** (epoch/2))

            adj_ent_in = sklearn.preprocessing.normalize(adj_ent_in, norm="l2", axis=1)
            adj_ent_out = sklearn.preprocessing.normalize(adj_ent_out, norm="l2", axis=1)
            adj_rel_in = sklearn.preprocessing.normalize(adj_rel_in, norm="l2", axis=1)
            adj_rel_out = sklearn.preprocessing.normalize(adj_rel_out, norm="l2", axis=1)



        #return np.concatenate((h_embed, adj_rel_in, adj_rel_out), axis=-1)
        return np.concatenate((h_embed, adj_ent_in, adj_ent_out), axis=-1)
        return np.concatenate((h_embed, adj_rel_in, adj_ent_in, adj_rel_out, adj_ent_out), axis=-1)
        return np.concatenate((h_embed, adj_rel_in, adj_ent_in,adj_rel_out,adj_ent_out), axis = -1)

    def justification(self,distance,left,right,topk=3,labels=None):
        #diag = [i for i in range(distance.shape[0])]
        #distance[diag,diag] = 1.0
        if (labels == None):
            label_topk_l = torch.topk(-A, topk,dim=-1).indices.numpy()

            l2r_pc = [x for x in range(len(left)) if right[label_topk_l[x,0]] == self.ill_idx_dic[left[x]]].__len__()
            r2l_pc = [x for x in range(len(right)) if left[label_topk_r[x,0]] == self.ill_idx_dic[right[x]]].__len__()
            print('l2r before justification is ' , l2r_pc)
            print('r2l before justification is ', r2l_pc)

            t2 = time.time();
            print('sorting lasts ' ,int(t2-t1),' seconds');
            t1 = time.time()
            gap = self.kg1_ins_ids.__len__()

            justification_l = np.zeros([self.ins_num,topk])
            justification_r = np.zeros([self.ins_num,topk])

            l_index = -np.ones([self.ins_num])
            r_index = -np.ones([self.ins_num])
            l_same =-np.ones([self.ins_num,topk])
            r_same =-np.ones([self.ins_num,topk])

            graph_list = [(h, t) for (h, r, t) in self.triple_idx]
            graph = {}
            for e in graph_list:
                graph[e] = 0
            for e in graph_list:
                graph[e] = 0
            for i in range(len(right)):
                r_same[right[i],:] = left[label_topk_r[i, :]]
                r_index[right[i]] = 1

            for i in range(len(left)):
                l_same[left[i],:] = right[label_topk_l[i, :]]
                l_index[left[i]] = 1


            for l in labels:
                r_same[l[0],:] = np.array(l[1]*topk)
                r_same[l[1],:] = np.array(l[0]*topk)
                l_same[l[0],:] = np.array(l[1]*topk)
                l_same[l[1],:] = np.array(l[0]*topk)

            for i, j in zip(range(topk), range(topk)):

                l_true_edges = [[h, t] for [h,t] in graph_list if (l_index[h] == 1 or l_index[t] == 1) and (l_same[h,i],l_same[t,j]) in graph]
                r_true_edges = [[h, t] for [h, t] in graph_list if (r_index[h] == 1 or r_index[t] == 1) and (r_same[h,i], r_same[t,j]) in graph]


                if(l_true_edges.__len__()>0):
                    l_true_edge_np = np.array(l_true_edges)
                    justification_l[l_true_edge_np[:, 0],i] += 1
                    justification_l[l_true_edge_np[:, 1],j] += 1
                if (r_true_edges.__len__() > 0):
                    r_true_edge_np = np.array(r_true_edges)
                    justification_r[r_true_edge_np[:, 0],i] += 1
                    justification_r[r_true_edge_np[:, 1],j] += 1

            t2 = time.time();
            print('mass justification lasts ' ,int(t2-t1),' seconds');
            t1 = time.time()

            justification_l= justification_l[left,:]
            justification_r= justification_r[right,:]

            best_label_l = np.argmax(justification_l, axis=-1)
            best_label_r = np.argmax(justification_r, axis=-1)
            #distance = np.ones(distance.shape)

            new_l2r_pc = [x for x in range(len(left)) if right[label_topk_l[x, best_label_l[x]]] == self.ill_idx_dic[left[x]]].__len__()
            new_r2l_pc = [x for x in range(len(right)) if left[label_topk_r[x, best_label_r[x]]] == self.ill_idx_dic[right[x]]].__len__()

            print('new l2r after justification is ' , new_l2r_pc , ' and improve :', new_l2r_pc - l2r_pc)
            print('new r2l after justification is ', new_r2l_pc, ' and improve :', new_r2l_pc - r2l_pc)

            for i in range(len(left)):
                if justification_l[i, best_label_l[i]] > 0:
                    distance[i,label_topk_l[i, best_label_l[i]]]= distance[i,label_topk_l[i, 0]] - 1/(100-n)

            for i in range(len(right)):
                if justification_r[i, best_label_r[i]] > 0:
                    distance[label_topk_r[i, best_label_r[i]],i ]= distance[label_topk_r[i, 0],i ] - 1/(100-n)

            t2 = time.time();
            print('compute new distance matrix lasts ' ,int(t2-t1),' seconds');

        return distance;



    def ReGAL (self,embeding,left,right):

        gap = self.ins_num//2

        #left = [i for i in range(gap)]
        #right = [i+gap for i in range(gap)]

        left_emb = embeding[left]
        right_emb = embeding[right]
        tree = KDTree(left_emb)
        dist, ind = tree.query(right_emb, k=1)

        same = {};
        for i in range(len(right_emb)):
            if(dist[i] < 0.1):
                same[left[int(ind[i])]] = right[i]

        edges = [[h,t] for (h, r, t) in self.triple_idx]+[[t,h] for (h, r, t) in self.triple_idx]

        justification = np.zeros(gap)
        for [e1,e2] in edges:
            if(e2 < gap and e1 < gap  ):
                if(e1 in same and e2 in same and [same[e1],same[e2]] in edges):
                    justification[e1] += 1
                    justification[e2] += 1

        idx = np.argsort(justification)
        candid_anchor = idx[-100:]

        return self.BFS(edges,candid_anchor ,same)

    def BFS(self,edges,start_idx,same):
        n_i = {}
        num_entity = self.ins_num
        for i in range(num_entity):
            n_i[i] = []
        for (e1,e2) in edges:
            n_i[e1].append(e2)
            n_i[e2].append(e1)
        embd = np.ones([num_entity,len(start_idx)])
        embd[:,:] = 1000
        ix = 0
        for x in start_idx:
            bfs = np.zeros([num_entity]);
            embd[x,ix] = 0
            embd[same[x], ix] = 0
            stack = [x,same[x]]
            bfs[x] = 1
            bfs[same[x]] = 1

            while(len(stack)>0):
                s = stack[0]
                stack = stack[1:]
                for n in n_i[s]:
                    embd[n,ix] = min(embd[n,ix],embd[s,ix]+1)
                    if(bfs[n]==0):
                        stack.append(n)
                        bfs[n] = 1

            ix += 1
        embd = sklearn.preprocessing.normalize(np.exp(-embd), norm="l2", axis=1)

        return embd

    def gen_sparse_graph_from_triples(self, triples, ins_num, with_r=False):
        edge_dict = {}
        in_nodes_dict = {}
        out_nodes_dict = {}
        in_rels_dict = {}
        out_rels_dict = {}

        for (h, r, t) in triples:
            if h != t:
                r1 = r + self.ins_num
                if (h, t) not in edge_dict:
                    edge_dict[(h, t)] = []
                    edge_dict[(t, h)] = []

                if (t, h) not in in_nodes_dict:
                    in_nodes_dict[(t, h)] = []
                if (t, r1) not in in_nodes_dict:
                    in_nodes_dict[(t, r1)] = []

                if (h, r1) not in out_nodes_dict:
                    out_nodes_dict[(h, r1)] = []
                if (h, t) not in out_nodes_dict:
                    out_nodes_dict[(h, t)] = []

                if (h, r1) not in out_nodes_dict:
                    in_rels_dict[(h,r1)] = []
                    in_rels_dict[(t,r1)] = []

                edge_dict[(h, t)].append(r)
                edge_dict[(t, h)].append(r1)



        if with_r:
            edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
            r_ij = [(r) for (h, t) in edge_dict for r in edge_dict[(h, t)]]

            in_nodes = [[h, t] for (h, t) in in_nodes_dict]
            out_nodes = [[h, t] for (h, t) in out_nodes_dict]
            in_rels = [[h, t] for (h, t) in in_rels_dict]
            out_rels = [[h, t] for (h, t) in out_rels_dict]

            edges = np.array(edges, dtype=np.int32)
            values = np.array(values, dtype=np.float32)
            r_ij = np.array(r_ij, dtype=np.float32)
            edges_dict = {'default': edges, 'in_nodes': in_nodes, 'out_nodes': out_nodes, 'in_rels': in_rels,
                          'rels' : in_rels + out_rels,
                          'out_rels': out_rels}
            return edges_dict, values, r_ij
        else:
            in_nodes = [[h, t] for (h, t) in in_nodes_dict]
            out_nodes = [[h, t] for (h, t) in out_nodes_dict]
            in_rels = [[h, t] for (h, t) in in_rels_dict]
            out_rels = [[h, t] for (h, t) in out_rels_dict ]


            if 1==1:
                #in_nodes += [[e, e] for e in range(ins_num + self.rel_num)]
                #out_nodes += [[e, e] for e in range(ins_num + self.rel_num)]
                in_rels += [[e, e] for e in range(ins_num,ins_num + self.rel_num)]
                out_rels += [[e, e] for e in range(ins_num,ins_num + self.rel_num)]

            edges = [[h, t] for (h, t) in edge_dict]
            values = [1 for (h, t) in edge_dict]
        # add self-loop

        edges += [[e, e] for e in range(ins_num)]
        values += [1 for e in range(ins_num)]
        edges = np.array(edges, dtype=np.int32)
        values = np.array(values, dtype=np.float32)

        edges_dict = {'default':edges,'in_nodes':in_nodes,'out_nodes':out_nodes,'in_rels':in_rels,'out_rels':out_rels}
        return edges_dict, values, None

    def share(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        new_triples = []
        for (h, r, t) in triples:
            if h in from_1_to_2_dict:
                h = from_1_to_2_dict[h]
            if t in from_1_to_2_dict:
                t = from_1_to_2_dict[t]
            new_triples.append((h, r, t))
        new_triples = list(set(new_triples))
        return new_triples

    def share_attr(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        new_triples = []
        for (h, r, t) in triples:
            if h in from_1_to_2_dict:
                h = from_1_to_2_dict[h]
            new_triples.append((h, r, t))
        new_triples = list(set(new_triples))
        return new_triples
    
    def swap(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        from_2_to_1_dict = dict(ill[:, ::-1])
        new_triples = []
        for (h, r, t) in triples:
            new_triples.append((h, r, t))
            if h in from_1_to_2_dict:
                new_triples.append((from_1_to_2_dict[h], r, t))
            if t in from_1_to_2_dict:
                new_triples.append((h, r, from_1_to_2_dict[t]))
            if h in from_2_to_1_dict:
                new_triples.append((from_2_to_1_dict[h], r, t))
            if t in from_2_to_1_dict:
                new_triples.append((h, r, from_2_to_1_dict[t]))
        new_triples = list(set(new_triples))
        return new_triples

    def swap_attr(self, triples, ill):
        from_1_to_2_dict = dict(ill)
        from_2_to_1_dict = dict(ill[:, ::-1])
        new_triples = []
        for (h, r, t) in triples:
            new_triples.append((h, r, t))
            if h in from_1_to_2_dict:
                new_triples.append((from_1_to_2_dict[h], r, t))
            if h in from_2_to_1_dict:
                new_triples.append((from_2_to_1_dict[h], r, t))

        new_triples = list(set(new_triples))
        return new_triples

    def __repr__(self):
        return self.__class__.__name__ + " dataset summary:" + \
            "\n\tins_num: " + str(self.ins_num) + \
            "\n\trel_num: " + str(self.rel_num) + \
            "\n\ttriple_idx: " + str(len(self.triple_idx)) + \
            "\n\trate: " + str(self.rate) + "\tval: " + str(self.val) + \
            "\n\till_idx(train/test/val): " + str(len(self.ill_idx)) + " = " + str(len(self.ill_train_idx)) + " + " + str(len(self.ill_test_idx)) + " + " + str(len(self.ill_val_idx)) + \
            "\n\tins_G_edges_idx: " + str(len(self.ins_G_edges_idx['default'])) + \
            "\n\t----------------------------- init_time: " + str(round(self.init_time, 3)) + "s"


if __name__ == '__main__':
    
    # TEST

    d = AlignmentData(share=False, swap=False,OpenEa = True)
    print(d)
    d = AlignmentData(share=True, swap=False)
    print(d)
    d = AlignmentData(share=False, swap=True)
    print(d)
    