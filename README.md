# Causality Inference
A common frustration in the industry, especially when it comes to getting business insights from tabular data, is that the most interesting questions (from their perspective) are often not answerable with observational data alone.

The causal graph is a central solution to the problem mentioned above, but it is often unknown, subject to personal knowledge and bias, or loosely connected to the available data. In statistics, econometrics, epidemiology, genetics and related disciplines, causal graphs (also known as path diagrams, causal Bayesian networks or DAGs) are probabilistic graphical models used to encode assumptions about the data-generating process. Causal graphs can be used for communication and for inference. Causal graphical models is a python module for describing and manipulating Causal Graphical Models and Structural Causal Models. Behind the scenes it is a light wrapper around the python graph library networkx, together with some CGM specific tools. It is currently in a very early stage of development.
Inferring the cause of a phenomenon is described as the identification of its cause or causes by establishing covariation of cause and effect, a time-order relationship with the cause predicting the effect, and the elimination of plausible alternative causes. Causal inference, on the other hand,  according to Wikipedia, is the process of determining the independent, actual effect of a particular phenomenon that is a component of a larger system. 

## Breast Cancer Causal Inference
Breast cancer is the cancer that forms in the cells of the breasts. This form of cancer occurs mostly in women and rarely in men
Some of it symptoms are a lump in the breast, bloody discharge from the nipple and changes in the shape or texture of the nipple or breast.

There are some features which are known to be causing breast cancer, as given in the data https://www.kaggle.com/uciml/breast-cancer-wisconsin-data. However, the general assumption is that there might be some underlying factors that causes breast cancer and might not be easily known from any observational or experimental data.
The main objective of the task is to highlight the possibility of that assumption and establish a ground truth.

This is an attempt to solve the problem using the follow methods:
<ul>
  <li>Perform a causal inference task using Pearl’s framework;</li>
  <li>Infer the causal graph from observational data and then validate the graph;</li>
  <li>Merge machine learning with causal inference;</li>
</ul>

## Conclusion
Insights derived from analysis carried out indicate that causal inference is highly needed to assess performance prediction on each individual component of study.
Also, the causes of breast cancer are well known, the analysis infers that there could be other underlying causes that are not apparent. 
