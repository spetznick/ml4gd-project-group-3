# ml4gd-project-group-3

## Python environment
We are using `Python 3.12.3`
Create a Python environment using `python -m virtualenv <env-name>`. Invoke the environment by `source <env-name>/bin/activate` and install the packages by `pip install -r requirements.txt`.

## Wind turbine farm data set

- Use the SCADA SWPF data set to formulate a problem on graph data
- The data set stems from  [Baidu KDD Cup 2022 challenge](https://aistudio.baidu.com/competition/detail/152/0/introduction)
- There might be other possible challenges to solve based on this data set
- A possible way to evaluate our problem is based on the challenge and provided in their [github](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/apps/wpf_baseline_gru)

## UK Smart meter data set
- Smart meters installed in around 5500 households in the London area between 2012 and 2014

### Downlaad
- The data set itself is too big to be commited and needs to be downloaded [here](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- The partitioned data sets should be unzipped and stored in `/uk-smart-meter-data`.
- A fraction of theses households have been informed about their day-ahead electricity price in order to adapt their usage of electricity
- A (major?) drawback of this data set is that we need to generate our own graph as the households are anonymised.