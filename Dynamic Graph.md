# Dynamic Graph



**Dynamic Graph Representation Learning with Neural Networks: A Survey** https://arxiv.org/abs/2304.05729

This paper aims at providing a review of problems and models related to dynamic graph learning. The various dynamic graph supervised learning settings are analysed and discussed.

According to our problem setting, we need to choose snapshot version, as the data is updated every 30 min.



Some thoughts:

1.  **Do we need node embedding or not?**

​		Many papers are focusing on the node embedding in dynamic graph learning. Considering we only have one 		feature,  it could be easier if we do not use embedding. 

​		If we would like to embed the nodes: check https://arxiv.org/pdf/2101.01229v1

2. **transductive or inductive?**

These two factors influence the learning setting and how we model the graph.



**Do We Really Need Complicated Model Architectures For Temporal Networks?** https://openreview.net/pdf?id=ayPPc0SyLv1

code:https://github.com/CongWeilin/GraphMixer

Propose a  technically simple architecture that consists of three components: 

1. a link-encoder that is only based on multi-layer perceptrons (MLP) to summarize the information from temporal links, 

2. a node-encoder that is only based on neighbor mean-pooling to summarize node information, and 3
3.  an MLP-based link classifier that performs link prediction based on the outputs of the encoders

Node-encoder is interesting for our project, which captures the node identity and node feature information via neighbor mean-pooling.  The node-info feature is computed based on the N-hop neighbour:

**si(t0) = x i + Mean{x j | vj ∈ N (vi ;t0 − T, t0)}**, basically si at timestamp T is computed by the mean of neighbors of node i from the period [t0 − T, t0]. However the method is heavily based on the edge construction.