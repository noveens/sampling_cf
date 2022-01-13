# On Sampling Collaborative Filtering Datasets

This repository contains the implementation of many popular sampling strategies, along with various explicit/implicit/sequential feedback recommendation algorithms. The code accompanies the paper ***"On Sampling Collaborative Filtering Datasets"*** [[ACM]](https://doi.org/10.1145/3488560.3498439) [[Public PDF]](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm22.pdf) where we compare the utility of different sampling strategies for preserving the performance of various recommendation algorithms.

We also provide code for `Data-Genie` which can automatically *predict* the performance of how good any sampling strategy will be for a given collaborative filtering dataset. We refer the reader to the full paper for more details. Kindly send me an email if you're interested in obtaining access to the pre-trained weights of `Data-Genie`.

If you find any module of this repository helpful for your own research, please consider citing the below WSDM'22 paper. Thanks!
```
@inproceedings{sampling_cf,
  author = {Noveen Sachdeva and Carole-Jean Wu and Julian McAuley},
  title = {On Sampling Collaborative Filtering Datasets},
  url = {https://doi.org/10.1145/3488560.3498439},
  booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  series = {WSDM '22},
  year = {2022}
}
```

**Code Author**: Noveen Sachdeva (nosachde@ucsd.edu)

---
### Setup
##### Environment Setup
```bash
$ pip install -r requirements.txt
```

##### Data Setup
Once you've correctly setup the python environments and downloaded the dataset of your choice (Amazon: http://jmcauley.ucsd.edu/data/amazon/), the following steps need to be run:

The following command will create the required data/experiment directories as well as download & preprocess the Amazon magazine and the MovieLens-100K datasets. Feel free to download more datasets from the following web-page http://jmcauley.ucsd.edu/data/amazon/ and adjust the `setup.sh` and `preprocess.py` files accordingly.

```bash
$ ./setup.sh
```

---
### How to train a model on a sampled/complete CF-dataset?
- Edit the `hyper_params.py` file which lists all config parameters, including what type of model to run. Currently supported models:

| Sampling Strategy | What is sampled? | Paper Link |
| --- | ------ | ------ |
| Random | Interactions |  |
| Stratified | Interactions |  |
| Temporal | Interactions |  |
| SVP-CF w/ MF | Interactions | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| SVP-CF w/ Bias-only | Interactions | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| SVP-CF-Prop w/ MF | Interactions | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| SVP-CF-Prop w/ Bias-only | Interactions | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| Random | Users |  |
| Head | Users |  |
| SVP-CF w/ MF | Users | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| SVP-CF w/ Bias-only | Users | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| SVP-CF-Prop w/ MF | Users | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| SVP-CF-Prop w/ Bias-only | Users | [LINK](https://doi.org/10.1145/3488560.3498439) & [LINK](https://openreview.net/forum?id=HJg2b0VYDr) |
| Centrality | Graph | [LINK](http://ilpubs.stanford.edu:8090/422/) |
| Random-Walk | Graph | [LINK](https://cs.stanford.edu/~jure/pubs/sampling-kdd06.pdf) |
| Forest-Fire | Graph | [LINK](https://www.cs.cornell.edu/home/kleinber/kdd05-time.pdf) |

- Finally, type the following command to run:
```bash
$ CUDA_VISIBLE_DEVICES=<SOME_GPU_ID> python main.py
```

- Alternatively, to train various possible recommendation algorithm on various CF datasets/subsets, please edit the configuration in `grid_search.py` and then run:
```bash
$ python grid_search.py
```

---
### How to train Data-Genie?
- Edit the `data_genie/data_genie_config.py` file which lists all config parameters, including what datasets/CF-scenarios/samplers etc. to train Data-Genie on

- Finally, use the following command to train Data-Genie:
```bash
$ python data_genie.py
```

### License
----

MIT
