# neural-template-gen

Code for [Learning Neural Templates for Text Generation](https://arxiv.org/abs/1808.10122) (Wiseman, Shieber, Rush; EMNLP 2018)

For questions/concerns/bugs please feel free to email swiseman[at]ttic.edu.

**N.B.** This code was tested with python 2.7 and pytorch 0.3.1.

## Data and Data Preparation

The E2E NLG Challenge data is available [here](http://www.macs.hw.ac.uk/InteractionLab/E2E/), and the preprocessed version of the data used for training is at [data/e2e_aligned.tar.gz](https://github.com/harvardnlp/neural-template-gen/blob/master/data/e2e_aligned.tar.gz). This preprocessed data uses the same database record preprocessing scheme applied by Sebastian Gehrmann in his [system](https://github.com/sebastianGehrmann/OpenNMT-py/tree/diverse_ensemble), and also annotates text spans that occur in the corresponding database. Code for annotating the data in this way is at [data/make_e2e_labedata.py](https://github.com/harvardnlp/neural-template-gen/blob/master/data/make_e2e_labedata.py).


The WikiBio data is available [here](https://github.com/DavidGrangier/wikipedia-biography-dataset), and the preprocessed version of the target-side data used for training is at [data/wb_aligned.tar.gz](https://github.com/harvardnlp/neural-template-gen/blob/master/data/wb_aligned.tar.gz). This target-side data is again preprocessed to annotate spans appearing in the corresponding database. Code for this annotation is at [data/make_wikibio_labedata.py](https://github.com/harvardnlp/neural-template-gen/blob/master/data/make_wikibio_labedata.py). The source-side data can be downloaded directly from the [WikiBio repo](https://github.com/DavidGrangier/wikipedia-biography-dataset), and we used it unchanged; in particular the `*.box` files become our `src_*.txt` files mentioned below.


The code assumes that each dataset lives in a directory containing `src_train.txt`, `train.txt`, `src_valid.txt`, and `valid.txt` files, and that if the files are from the WikiBio dataset the directory name will contain the string `wiki`.

## Training
The four trained models mentioned in the paper can be downloaded [here](https://drive.google.com/drive/folders/1iv71Oq7cmXRY6h2jn0QzlYbbr0GwHCfA?usp=sharing). The commands for retraining the models are given below.

Assuming your E2E data is in `data/labee2e/`, you can train the non-autoregressive model as follows

```
python chsmm.py -data data/labee2e/ -emb_size 300 -hid_size 300 -layers 1 -K 55 -L 4 -log_interval 200 -thresh 9 -emb_drop -bsz 15 -max_seqlen 55 -lr 0.5 -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 5 -mlpinp -onmt_decay -cuda -seed 1818 -save models/chsmm-e2e-300-55-5.pt
```

and the autoregressive model as follows.

```
python chsmm.py -data data/labee2e/ -emb_size 300 -hid_size 300 -layers 1 -K 55 -L 4 -log_interval 200 -thresh 9 -emb_drop -bsz 15 -max_seqlen 55 -lr 0.5 -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 5 -mlpinp -onmt_decay -cuda -seed 1111 -save models/chsmm-e2e-300-55-5-far.pt -ar_after_decay
```


Assuming your WikiBio data is in `data/labewiki`, you can train the non-autoregressive model as follows

```
python chsmm.py -data data/labewiki/ -emb_size 300 -hid_size 300 -layers 1 -K 45 -L 4 -log_interval 1000 -thresh 29 -emb_drop -bsz 5 -max_seqlen 55 -lr 0.5 -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 3 -mlpinp -onmt_decay -cuda -save models/chsmm-wiki-300-45-3.pt
```

and the autoregressive model as follows.

```
python chsmm.py -data data/labewiki/ -emb_size 300 -hid_size 300 -layers 1 -K 45 -L 4 -log_interval 1000 -thresh 29 -emb_drop -bsz 5 -max_seqlen 55 -lr 0.5 -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 3 -mlpinp -onmt_decay -cuda -save models/chsmm-wiki-300-45-3-war.pt -ar_after_decay -word_ar
```

The above scripts will also attempt to save to a `models/` directory (which must be created first). Also see [chsmm.py](https://github.com/harvardnlp/neural-template-gen/blob/master/chsmm.py) for additional training and model options.

**N.B.** training is somewhat sensitive to the random seed, and it may be necessary to try several seeds in order to get the best performance.


## Viterbi Segmentation/Template Extraction

Once you've trained a model, you can use it to compute the Viterbi segmentation of the training data, which we use to extract templates. A gzipped tarball containing Viterbi segmentations corresponding to the four models above can be downloaded [here](https://drive.google.com/file/d/1ON4ROs_coDNmVt3-JON4wK1Kc_NkIV2M/view?usp=sharing).

You can rerun the segmentation for the non-autoregressive E2E model as follows

```
python chsmm.py -data data/labee2e/ -emb_size 300 -hid_size 300 -layers 1 -K 55 -L 4 -log_interval 200 -thresh 9 -emb_drop -bsz 16 -max_seqlen 55 -lr 0.5  -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 5 -mlpinp -onmt_decay -cuda -load models/e2e-55-5.pt -label_train | tee segs/seg-e2e-300-55-5.txt
```

and for the autoregressive one as follows.

```
python chsmm.py -data data/labee2e/ -emb_size 300 -hid_size 300 -layers 1 -K 60 -L 4 -log_interval 200 -thresh 9 -emb_drop -bsz 16 -max_seqlen 55 -lr 0.5  -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 1 -mlpinp -onmt_decay -cuda -load models/e2e-60-1-far.pt -label_train -ar_after_decay | tee segs/seg-e2e-300-60-1-far.txt
```

You can rerun the segmentation for the non-autoregressive WikiBio model as follows

```
python chsmm.py -data data/labewiki/ -emb_size 300 -hid_size 300 -layers 1 -K 45 -L 4 -log_interval 200 -thresh 29 -emb_drop -bsz 16 -max_seqlen 55 -lr 0.5  -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 3 -mlpinp -onmt_decay -cuda -load models/wb-45-3.pt -label_train | tee segs/seg-wb-300-45-3.txt
```

and for the autoregressive one as follows.

```
python chsmm.py -data data/labewiki/ -emb_size 300 -hid_size 300 -layers 1 -K 45 -L 4 -log_interval 200 -thresh 29 -emb_drop -bsz 16 -max_seqlen 55 -lr 0.5  -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 3 -mlpinp -onmt_decay -cuda -load models/wb-45-3-war.pt -label_train | tee segs/seg-wb-300-45-3-war.txt
```

The above scripts write the MAP segmentations (in text) to standard out. Above, they have been redirected to a `segs/` directory.

### Examining and Extracting Templates
The [template_extraction.py](https://github.com/harvardnlp/neural-template-gen/blob/master/template_extraction.py) script can be used to extract templates from the segmentations produced as above, and to look at them. In particular, `extract_from_tagged_data()` returns the most common templates, and mappings from these templates to sentences, and from states to phrases. This script is also used in generation (see below).


## Generation
Once a model has been trained and the MAP segmentations created, we can generate by limiting to (for instance) the top 100 extracted templates.

The following command will generate on the E2E validation set using the autoregressive model:

```
python chsmm.py -data data/labee2e/ -emb_size 300 -hid_size 300 -layers 1 -dropout 0.3 -K 60 -L 4 -log_interval 100 -thresh 9 -lr 0.5 -sep_attn -unif_lenps -emb_drop -mlpinp -onmt_decay -one_rnn -max_pool -gen_from_fi data/labee2e/src_uniq_valid.txt -load models/e2e-60-1-far.pt -tagged_fi segs/seg-e2e-60-1-far.txt -beamsz 5 -ntemplates 100 -gen_wts '1,1' -cuda -min_gen_tokes 0 > gens/gen-e2e-60-1-far.txt
```

The following command will generate on the WikiBio test using the autoregressive model:
```
python chsmm.py -data data/labewiki/ -emb_size 300 -hid_size 300 -layers 1 -K 45 -L 4 -log_interval 1000 -thresh 29 -emb_drop -bsz 5 -max_seqlen 55 -lr 0.5 -sep_attn -max_pool -unif_lenps -one_rnn -Kmul 3 -mlpinp -onmt_decay -cuda -gen_from_fi wikipedia-biography-dataset/test/test.box -load models/wb-45-3-war.pt -tagged_fi segs/seg-wb-300-45-3-war.txt -beamsz 5 -ntemplates 100 -gen_wts '1,1' -cuda -min_gen_tokes 20 > gens/gen-wb-45-3-war.txt
```

Generations from the other models can be obtained analogously, by substituting in the correct arguments for `-data` (path to data directory), `-gen_from_fi` (the source file from which to generate), `-load` (path to the saved model), and `-tagged_fi` (path to the MAP segmentations under the corresponding model). See [chsmm.py](https://github.com/harvardnlp/neural-template-gen/blob/master/chsmm.py) for additional generation options.


**N.B.** The format of the generations is: `<generation>|||<segmentation>`, where `<segmentation>` provides the segmentation used in generating. As such, all the text beginning with '|||' should be stripped off before evaluating the generations.
