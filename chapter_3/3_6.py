'''
@Filename: 
@Author: shifaqiang
@Email: 14061115@buaa.edu.cn
@Github: https://github.com/luoboganer
@Date: 2019-05-14 11:53:57
@LastEditors: shifaqiang
@LastEditTime: 2019-05-15 10:20:08
@Software: Visual Studio Code
@Description: 
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../style/elegant.mplstyle')
from skimage import io,segmentation,color,future
import numpy as np
import networkx as nx
from scipy import ndimage as ndi

def add_edge_filter(values,graph):
    center=values[len(values)//2]
    for neighbor in values:
        if neighbor!=center and not graph.has_edge(center,neighbor):
            graph.add_edge(center,neighbor)
    return 0.0 # return value just for satisfying the requirement of generic_filter

def build_rag(labels,image):
    graph=nx.Graph()
    footprint=ndi.generate_binary_structure(labels.ndim,connectivity=1)
    _=ndi.generic_filter(labels,add_edge_filter,footprint=footprint,mode='nearest',extra_arguments=(graph,))
    for n in graph:
        graph.node[n]['total color']=np.zeros(3,np.double)
        graph.node[n]['pixel count']=0
    for index in np.ndindex(labels.shape):
        n=labels[index]
        graph.node[n]['total color']+=image[index]
        graph.node[n]['pixel count']+=1
    return graph

def threshold_graph(graph,t):
    to_remove=[(u,v) for (u,v,d) in graph.edges(data=True) if d['weight']>t]
    graph.remove_edges_from(to_remove)

def rag_segmentation_tiger(seg,image):
    graph=build_rag(seg,image)
    for n in graph:
        node=graph.node[n]
        node['mean']=node['total color']/node['pixel count']
    for u,v in graph.edges:
        d=graph.node[u]['mean']-graph.node[v]['mean']
        graph[u][v]['weight']=np.linalg.norm(d)
    threshold_graph(graph,80)
    map_array=np.zeros(np.max(seg)+1,np.int)
    for i,segment in enumerate(nx.connected_components(g)):
        for initial in segment:
            map_array[int(initial)]=i
    segmented=map_array[seg]
    tiger_slic_seg=color.label2rgb(segmented,image)
    io.imsave('./3_6_rag_tiger.png',tiger_slic_seg)

if __name__ == "__main__":
    # url= https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg
    filename_tiger='../data/108073.jpg'
    tiger=io.imread(filename_tiger)
    seg=segmentation.slic(tiger,n_segments=30,compactness=40.0,enforce_connectivity=True,sigma=3)
    tiger_slic=color.label2rgb(seg,tiger)*255
    tiger_slic=tiger_slic.astype(np.uint8)
    io.imsave('./3_6_slic.png',tiger_slic)
    g=future.graph.rag_mean_color(tiger,seg)
    future.graph.show_rag(seg,g,tiger,edge_cmap='YlGnBu')
    plt.savefig('./3_6_rag.png')
    rag_segmentation_tiger(seg,tiger)