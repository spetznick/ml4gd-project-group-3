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





