"""Split only internally disconnected communities; retain every node and link.

For nonnegative bipartite weights and gamma>=0, splitting a community into
components of its induced observed graph leaves its observed internal amount
unchanged and removes nonnegative cross-component strength-product penalties.
Therefore Q cannot decrease (apart from numerical rounding). This is a specified
refinement, not a claim that the whole optimizer is the Leiden algorithm.
"""
import numpy as np
from numba import njit


@njit(cache=True,nogil=True)
def component_labels(wallet_indptr,wallet_events,wallet_labels,event_labels):
    nw=len(wallet_labels);ne=len(event_labels);n=nw+ne
    parent=np.arange(n,dtype=np.int32);size=np.ones(n,dtype=np.int32)
    for u in range(nw):
        for j in range(wallet_indptr[u],wallet_indptr[u+1]):
            v=wallet_events[j]
            if wallet_labels[u]!=event_labels[v]:continue
            a=u;b=nw+v
            while parent[a]!=a:
                parent[a]=parent[parent[a]];a=parent[a]
            while parent[b]!=b:
                parent[b]=parent[parent[b]];b=parent[b]
            if a!=b:
                if size[a]<size[b]:a,b=b,a
                parent[b]=a;size[a]+=size[b]
    root_to_label=np.full(n,-1,dtype=np.int32)
    labels=np.empty(n,dtype=np.int32);next_label=0
    for i in range(n):
        p=i
        while parent[p]!=p:
            parent[p]=parent[parent[p]];p=parent[p]
        if root_to_label[p]<0:
            root_to_label[p]=next_label;next_label+=1
        labels[i]=root_to_label[p]
    return labels[:nw].copy(),labels[nw:].copy(),next_label


def split_disconnected(wallet_indptr,wallet_events,wallet_labels,event_labels):
    """Return {wallet_labels,event_labels,n_components,extra_components}.

    Inputs must have matching valid sparse endpoints and complete memberships;
    the storage/optimizer caller owns validation. Isolates are retained.
    """
    old_count=len(np.unique(np.concatenate((wallet_labels,event_labels))))
    w,e,n=component_labels(wallet_indptr,wallet_events,wallet_labels,event_labels)
    return {'wallet_labels':w,'event_labels':e,'n_components':int(n),
            'extra_components':int(n-old_count),'nodes_excluded':0,'links_excluded':0}
