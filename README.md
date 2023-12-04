# Improving Aspect-Based Sentiment with SRL

This is the repository for the code of our paper: 

## Improving Aspect-Based Sentiment with End-to-End Semantic Role Labeling Model

Accepted to [RANLP 2023](http://ranlp.org/ranlp2023/) conference.

Usage:
--------


### Train SRL-only model

```
python3 run.py  --model_name google/electra-small-discriminator --use_automodel --dataset_name cs_srl_e2e --task SRL 
--solution_type_cat NLI_M --epoch_num 20 --use_custom_model --max_seq_len 200 --lr 1e-4 --dataset_lang en --end2end--srl_official_eval --save_model
```

### Train joined model concat-conv
```
python3 run.py --use_automodel --model_name google/electra-small-discriminator --dataset_name semeval2014_en 
--task CAT --solution_type_cat NLI_B  --use_custom_model --injection_mode concat-convolution --use_pre_trained_srl_model 
--pre_trained_srl_model_path ./data/local_models/electra-small-discriminator_SRL-Pre-trained --max_seq_len 256 --dataset_lang en
```

### Joined model multi-task

```
python3 run.py --use_automodel --model_name google/electra-small-discriminator --dataset_name en_absa_srl_dataset
 --task CAT --solution_type_cat NLI_B --injection_mode multi-task --use_custom_model --dataset_lang en
```

to see a detailed description of all parameters please run:

```
python3 run.py --help
```


Setup:
--------

Create conda enviroment

1) ### Clone github repository 
   ```
   git clone git@github.com:pauli31/srl-aspect-based-sentiment.git
   ```
2) ### Setup conda
    
3) ### Setup Data
   #### SRL English:
   1) Download OntoNotes v5.0 corpus and convert it to conll format according to instructions here: https://cemantix.org/data/ontonotes.html
   2) Place train, dev, and test splits into folder data/datasets/srl/en/

   
   
   
 

License:
--------
The code can be freely used for academic and research purposes.
It is strictly prohibited to use it for any commercial purpose.

Publication:
--------

If you use our software for academic research, please cite our [paper](https://arxiv.org/abs/2307.14785)

```
@inproceedings{priban-steinberger-2022-czech,
    title = "Improving Aspect-Based Sentiment with End-to-End Semantic Role Labeling Model",
    author = "P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
      Pra{\v{z}}{\'a}k, Ond{\v{r}}ej",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2023)",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/",
    pages = "",
}
```

Official proceedings citation will follow soon.

Contact:
--------
pribanp@kiv.zcu.cz, ondfa@kiv.zcu.cz

[http://nlp.kiv.zcu.cz](http://nlp.kiv.zcu.cz)

