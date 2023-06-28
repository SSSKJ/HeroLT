import torch
import numpy as np
from scipy.spatial.distance import pdist,squareform
import random

def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_new = None

    cnt = 0
    for i in range(c_largest):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        if chosen.shape[0] > 0 and cnt < im_class_num:
            cnt += 1
            num = int(chosen.shape[0]*portion)
            if portion == 0:
                avg_number = int(idx_train.shape[0] / (c_largest + 1))
                c_portion = int(avg_number/chosen.shape[0])
                num = chosen.shape[0]
            else:
                c_portion = 1

            for j in range(c_portion):
                chosen = chosen[:num]

                chosen_embed = embed[chosen,:]
                distance = squareform(pdist(chosen_embed.cpu().detach()))
                np.fill_diagonal(distance,distance.max()+100)

                idx_neighbor = distance.argmin(axis=-1) # Equation 3

                interp_place = random.random()
                new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place # Equation 4


                new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(c_largest-i)
                idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
                idx_train_append = idx_train.new(idx_new)

                embed = torch.cat((embed,new_embed), 0)
                labels = torch.cat((labels,new_labels), 0)
                idx_train = torch.cat((idx_train,idx_train_append), 0)

                ## The generated edges are only from those that were originally edges of the sampled nodes and the nearest neighbor of the sampled nodes
                if adj is not None:
                    adj[adj != 0] = 1
                    if adj_new is None:
                        # adj_new = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0)) #?
                        adj_new = adj.new(torch.clamp_(adj[chosen, :] + adj[chosen[idx_neighbor], :], min=0.0, max=1.0))  # ?
                    else:
                        # temp = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                        temp = adj.new(torch.clamp_(adj[chosen, :] + adj[chosen[idx_neighbor], :], min=0.0, max=1.0))  # ?
                        adj_new = torch.cat((adj_new, temp), 0)

    # return embed, labels, idx_train

    ##
    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train