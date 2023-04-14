# Awesome-Multimodal-Video [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)



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

  * **VIOLETv2 (EmpiricalMVM)**: "An Empirical Study of End-to-End Video-Language Transformers with Masked Visual Modeling", CVPR 2023.
    <br/>[[Paper](https://arxiv.org/abs/2209.01540)][[Code](https://github.com/tsujuifu/pytorch_empirical-mvm)] #Microsoft

  * **LAVENDER**: "LAVENDER: Unifying Video-Language Understanding as Masked Language Modeling", CVPR 2023.
    <br/>[[Paper](https://arxiv.org/abs/2206.07160)][[Code](https://github.com/microsoft/LAVENDER)] #Microsoft

  * **Flamingo**: "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022.
    <br/>[[Paper](https://arxiv.org/abs/2204.14198)] #DeepMind

  * **ALPRO**: "Align and Prompt: Video-and-Language Pre-training with Entity Prompts", CVPR 2022.
    <br/>[[Paper](https://arxiv.org/abs/2112.09583)][[Code](https://github.com/salesforce/ALPRO)] #Salesforce

  * **MERLOT Reserve**: "MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound", CVPR 2022.
    <br/>[[Paper](http://arxiv.org/abs/2201.02639)][[Code](https://github.com/rowanz/merlot_reserve)][[Website](https://rowanzellers.com/merlotreserve/)][[Demo](https://merlot.apps.allenai.org/)] #AI2

  * **VL-Adapter**: "VL-ADAPTER: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks", CVPR 2022.
    <br/>[[Paper](https://arxiv.org/abs/2112.06825)][[Code](https://github.com/ylsung/VL_adapter)] #UNC


  * **MERLOT**: "MERLOT: Multimodal Neural Script Knowledge Models", NeurIPS 2021.
    <br/>[[Paper](https://arxiv.org/abs/2106.02636)][[Code](https://github.com/rowanz/merlot)][[Website](https://rowanzellers.com/merlot/)] #AI2

  * **VIOLET**: "VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling", arXiv 2021.
    <br/>[[Paper](https://arxiv.org/abs/2111.12681)][[Code](https://github.com/tsujuifu/pytorch_violet)] #Microsoft

  * **HERO**: "HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training", EMNLP 2020.
    <br/>[[Paper](https://arxiv.org/abs/2005.00200)][[Code](https://github.com/linjieli222/HERO)] #Microsoft

  * **UniVL**: "UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation", arXiv 2020.
    <br/>[[Paper](https://arxiv.org/abs/2002.06353)][[Code](https://github.com/microsoft/UniVL)] #Microsoft


## Video Retrieval

  * **FiT**: "Frozen in Time: ️A Joint Video and Image Encoder for End to End Retrieval", ICCV 2021.
    <br/>[[Paper](https://arxiv.org/abs/2104.00650)][[Code](https://github.com/m-bain/frozen-in-time)][[Website](https://www.robots.ox.ac.uk/~vgg/research/frozen-in-time)][[Demo](http://meru.robots.ox.ac.uk/frozen-in-time)] #Oxford

## Video Question Answering

  * **FrozenBiLM**: "Zero-Shot Video Question Answering via Frozen Bidirectional Language Models", NeurIPS 2022.
    <br/>[[Paper](https://arxiv.org/abs/2206.08155)][[Code](https://github.com/antoyang/FrozenBiLM)][[Website](https://antoyang.github.io/frozenbilm.html)][[Poster](https://antoyang.github.io/slides/frozenbilm-neurips-poster.pdf)][[Slides](https://antoyang.github.io/slides/frozenbilm-neurips.pdf)] #Inria


  * **JustAsk**: "Just Ask: Learning to Answer Questions from Millions of Narrated Videos"
    <br/>[[Paper](https://arxiv.org/abs/2012.00451)][[Code](https://github.com/antoyang/just-ask)][[Website](https://antoyang.github.io/just-ask.html)][[Demo](http://videoqa.paris.inria.fr/)][[Poster](https://antoyang.github.io/slides/just-ask-iccv-poster.pdf)][[Slides](https://antoyang.github.io/slides/just-ask-iccv.pdf)][[Oral](https://youtu.be/jzXdRT5W3C4?t=17280)] #Inria

## Video Captioning

  * **Video ChatCaptioner**: "Video ChatCaptioner: Towards the Enriched Spatiotemporal Descriptions", arXiv 2023.
    <br/>[[Paper](https://arxiv.org/abs/2304.04227)][[Code](https://github.com/Vision-CAIR/ChatCaptioner)] #KAUST

  * **Vid2Seq**: "Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning", CVPR 2023.
    <br/>[[Paper](https://arxiv.org/abs/2302.14115)][[Code](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq)][[Website](https://antoyang.github.io/vid2seq.html)][[Blog](https://ai.googleblog.com/2023/03/vid2seq-pretrained-visual-language.html)] #Google

  * **MV-GPT**: "End-to-end Generative Pretraining for Multimodal Video Captioning", CVPR 2022.
    <br/>[[Paper](https://arxiv.org/abs/2201.08264)] #Google

  * **SwinBERT**: "SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning", CVPR 2022.
    <br/>[[Paper](https://arxiv.org/abs/2111.13196)][[Code](https://github.com/microsoft/SwinBERT)] #Microsoft

# Datasets and SOTA

## Large-scale Video Language Dataset

  * **WebVid-10M**: "Frozen in Time: A Joint Video and Image Encoder for End to End Retrieval", ICCV 2021.
  <br/>[[Paper](https://arxiv.org/abs/2104.00650)][[Code](https://github.com/m-bain/webvid)][[Website](https://m-bain.github.io/webvid-dataset)] #Oxford

  * **HowTo100M**: "HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips", ICCV 2019.
  <br/>[[Paper](https://arxiv.org/abs/1906.03327)][[Code](https://github.com/antoine77340/howto100m)][[Website](https://www.di.ens.fr/willow/research/howto100m)] #Inria

## Downstream Tasks

  * **QVHighlights**: "QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries"

  * **VidSitu**: "Visual Semantic Role Labeling for Video Understanding"

  * **How2QA**: "HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training"

  * **How2R**: "HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training"
 
  * **TVC**: "TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval"

  * **TVR**: "TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval"

  * **TVQA**: "TVQA: Localized, Compositional Video Question Answering"


    | Model                                     | Val (Acc)                   | Test (Acc)                 |
    | ----------------------------------------  | ---------------------------:| --------------------------:|
    | MERLOT                                    |                        78.7 |                       78.4 |
    | MERLOT Reserve                            |                        86.5 |                       86.1 |



  * **MSR-VTT**: "MSR-VTT: A Large Video Description Dataset for Bridging Video and Language"

  * **YouCook2**: "Towards Automatic Learning of Procedures from Web Instructional Videos"

  * **Charades**: "Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding"

  * **DiDeMo**: "Localizing Moments in Video with Natural Language"