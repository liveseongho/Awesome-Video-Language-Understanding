# Awesome-Video-Language-Understanding [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

We introduce recent works on Awesome Video Language Understanding. 

To access full version, click [here](https://github.com/liveseongho/Awesome-Video-Language-Understanding/blob/main/README_full.md).

## Table of Contents

* [Main](#main)
  * [Video Language Transformers](#video-language-transformers)
  * [Video Retrieval](#video-retrieval)
  * [Video Question Answering](#video-question-answering)
  * [Video Captioning](#video-captioning)
* [Datasets and SOTA](#datasets-and-sota)
   * [Large-scale Video Language Dataset](#large-scale-video-language-dataset)
   * [Downstream Tasks](#downstream-tasks)

# Main

## Video Language Transformers

  * **VIOLETv2 (EmpiricalMVM)** [[Paper](https://arxiv.org/abs/2209.01540)][[Code](https://github.com/tsujuifu/pytorch_empirical-mvm)] @Microsoft
  <br/> An Empirical Study of End-to-End Video-Language Transformers with Masked Visual Modeling (CVPR 2023)
    
  * **LAVENDER** [[Paper](https://arxiv.org/abs/2206.07160)][[Code](https://github.com/microsoft/LAVENDER)] @Microsoft
    <br/> LAVENDER: Unifying Video-Language Understanding as Masked Language Modeling (CVPR 2023)

  * **Flamingo**[[Paper](https://arxiv.org/abs/2204.14198)] @DeepMind
    <br/> Flamingo: a Visual Language Model for Few-Shot Learning (NeurIPS 2022)

  * **ALPRO** [[Paper](https://arxiv.org/abs/2112.09583)][[Code](https://github.com/salesforce/ALPRO)] @Salesforce
    <br/> Align and Prompt: Video-and-Language Pre-training with Entity Prompts (CVPR 2022)

  * **VL-Adapter** [[Paper](https://arxiv.org/abs/2112.06825)][[Code](https://github.com/ylsung/VL_adapter)] @UNC
    <br/> VL-ADAPTER: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks (CVPR 2022)

  * **VIOLET** [[Paper](https://arxiv.org/abs/2111.12681)][[Code](https://github.com/tsujuifu/pytorch_violet)] @Microsoft
    <br/> VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling (arXiv 2021)

  * **HERO** [[Paper](https://arxiv.org/abs/2005.00200)][[Code](https://github.com/linjieli222/HERO)] @Microsoft
    <br/> HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training (EMNLP 2020)

  * **UniVL** [[Paper](https://arxiv.org/abs/2002.06353)][[Code](https://github.com/microsoft/UniVL)] @Microsoft
    <br/> UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation (arXiv 2020)
    
  * **InterVideo** [[Paper](https://arxiv.org/abs/2212.03191)][[Code](https://github.com/OpenGVLab/InternVideo)] @OpenGVLab
    <br/> InternVideo: General Video Foundation Models via Generative and Discriminative Learning (arXiv 2022)

## Video Retrieval

  * **FiT** [[Paper](https://arxiv.org/abs/2104.00650)][[Code](https://github.com/m-bain/frozen-in-time)][[Website](https://www.robots.ox.ac.uk/~vgg/research/frozen-in-time)][[Demo](http://meru.robots.ox.ac.uk/frozen-in-time)] @Oxford
    <br/> Frozen in Time: ️A Joint Video and Image Encoder for End to End Retrieval (ICCV 2021)

## Video Question Answering

  * **FrozenBiLM** [[Paper](https://arxiv.org/abs/2206.08155)][[Code](https://github.com/antoyang/FrozenBiLM)][[Website](https://antoyang.github.io/frozenbilm.html)][[Poster](https://antoyang.github.io/slides/frozenbilm-neurips-poster.pdf)][[Slides](https://antoyang.github.io/slides/frozenbilm-neurips.pdf)] @Inria
    <br/> Zero-Shot Video Question Answering via Frozen Bidirectional Language Models (NeurIPS 2022)

  * **MERLOT Reserve** [[Paper](http://arxiv.org/abs/2201.02639)][[Code](https://github.com/rowanz/merlot_reserve)][[Website](https://rowanzellers.com/merlotreserve/)][[Demo](https://merlot.apps.allenai.org/)] @AI2
    <br/> MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound (CVPR 2022)

  * **MERLOT** [[Paper](https://arxiv.org/abs/2106.02636)][[Code](https://github.com/rowanz/merlot)][[Website](https://rowanzellers.com/merlot/)] @AI2 
    <br/> MERLOT: Multimodal Neural Script Knowledge Models (NeurIPS 2021)

  * **JustAsk** [[Paper](https://arxiv.org/abs/2012.00451)/[Journal](https://arxiv.org/abs/2205.05019v2)][[Code](https://github.com/antoyang/just-ask)][[Website](https://antoyang.github.io/just-ask.html)][[Demo](http://videoqa.paris.inria.fr/)][[Poster](https://antoyang.github.io/slides/just-ask-iccv-poster.pdf)][[Slides](https://antoyang.github.io/slides/just-ask-iccv.pdf)][[Oral](https://youtu.be/jzXdRT5W3C4?t=17280)] @Inria 
  <br/> Just Ask: Learning to Answer Questions from Millions of Narrated Videos (ICCV 2021)
  <br/> Learning to Answer Visual Questions from Web Videos (TPAMI 2022)
  

## Video Captioning

  * **Video ChatCaptioner** [[Paper](https://arxiv.org/abs/2304.04227)][[Code](https://github.com/Vision-CAIR/ChatCaptioner)] @KAUST
    <br/> Video ChatCaptioner: Towards the Enriched Spatiotemporal Descriptions (arXiv 2023)

  * **Vid2Seq** [[Paper](https://arxiv.org/abs/2302.14115)][[Code](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq)][[Website](https://antoyang.github.io/vid2seq.html)][[Blog](https://ai.googleblog.com/2023/03/vid2seq-pretrained-visual-language.html)] @Google
    <br/> Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning (CVPR 2023)

  * **MV-GPT** [[Paper](https://arxiv.org/abs/2201.08264)] @Google
    <br/> End-to-end Generative Pretraining for Multimodal Video Captioning (CVPR 2022)

  * **SwinBERT** [[Paper](https://arxiv.org/abs/2111.13196)][[Code](https://github.com/microsoft/SwinBERT)] @Microsoft
    <br/> SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning (CVPR 2022)

# Datasets and SOTA

## Large-scale Video Language Dataset

  * **WebVid-10M** [[Paper](https://arxiv.org/abs/2104.00650)][[Code](https://github.com/m-bain/webvid)][[Website](https://m-bain.github.io/webvid-dataset)] @Oxford
  <br/> Frozen in Time: A Joint Video and Image Encoder for End to End Retrieval (ICCV 2021)

  * **HowTo100M** [[Paper](https://arxiv.org/abs/1906.03327)][[Code](https://github.com/antoine77340/howto100m)][[Website](https://www.di.ens.fr/willow/research/howto100m)] @Inria
  <br/> HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips (ICCV 2019)

## Downstream Tasks

  * **STAR** [[Paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/5ef059938ba799aaa845e1c2e8a762bd-Abstract-round2.html)][[Code](https://github.com/csbobby/STAR_Benchmark)][[Website](https://bobbywu.com/STAR)][[PaperswithCode](https://paperswithcode.com/dataset/star-1)] @MIT-IBM
    <br/> STAR: A Benchmark for Situated Reasoning in Real-World Videos (NeurIPS 2021)

  * **TVQA** [[Paper](https://arxiv.org/abs/1809.01696)][[Code](https://github.com/jayleicn/TVQA)][[Website](http://tvqa.cs.unc.edu)][[PapersWithCode](https://paperswithcode.com/dataset/tvqa)] @UNC
    <br/> TVQA: Localized, Compositional Video Question Answering (EMNLP 2018)
    
  * **YouCook2** [[Paper](https://arxiv.org/abs/1703.09788)][[Website](http://youcook2.eecs.umich.edu)][[PapersWithCode](https://paperswithcode.com/dataset/youcook2)] @UMich
  <br/> Towards Automatic Learning of Procedures from Web Instructional Videos (AAAI 2018)

  * **ActivityNet Captions** [[Paper](https://arxiv.org/abs/1705.00754)][[Code](https://github.com/ranjaykrishna/densevid_eval)][[Website](https://cs.stanford.edu/people/ranjaykrishna/densevid)][[PapersWithCode](https://paperswithcode.com/dataset/activitynet-captions)] @Stanford
  <br/> Dense-Captioning Events in Videos (ICCV 2017)

  * **Charades-STA** [[Paper](https://arxiv.org/abs/1705.02101)][[Code](https://github.com/jiyanggao/TALL)][[PapersWithCode](https://paperswithcode.com/dataset/charades-sta)] @USC
  <br/> Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding (ICCV 2017)

  * **DiDeMo** [[Paper](https://arxiv.org/abs/1708.01641)][[Code](https://github.com/LisaAnne/LocalizingMoments)][[PapersWithCode](https://paperswithcode.com/dataset/didemo)] @Adobe
  <br/> Localizing Moments in Video with Natural Language (ICCV 2017)

  * **MSVD** [[Paper](https://www.microsoft.com/en-us/research/publication/collecting-highly-parallel-data-for-paraphrase-evaluation)][[PapersWithCode](https://paperswithcode.com/dataset/msvd)] @Microsoft
  <br/> Collecting Highly Parallel Data for Paraphrase Evaluation (ACL 2017)

  * **LSMDC** [[Paper](https://arxiv.org/abs/1605.03705)][[Website](https://sites.google.com/site/describingmovies)][[PapersWithCode](https://paperswithcode.com/dataset/lsmdc)] @MPII
  <br/> Movie Description (IJCV 2017)

  * **MSR-VTT** [[Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf)][[PapersWithCode](https://paperswithcode.com/dataset/msr-vtt)] @Microsoft
  <br/> MSR-VTT: A Large Video Description Dataset for Bridging Video and Language (CVPR 2016)

  * **MPII-MD** [[Paper](https://arxiv.org/abs/1501.02530)][[Website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset)][[PapersWithCode](https://paperswithcode.com/dataset/lsmdc)] @MPII
  <br/> A Dataset for Movie Description (CVPR 2015)
