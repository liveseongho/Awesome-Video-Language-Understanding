# awesome-Video-Language-Understanding [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Maybe awesome?

If you find missed papers, please open issues or pull requests (recommended).

## Table of Contents

* [Main](#main)
  * [Video Language Transformers](#video-language-transformers)
  * [Video Retrieval](#video-retrieval)
  * [Video Question Answering](#video-question-answering)
  * [Video Captioning](#video-captioning)
  * [Others](#others)
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

  * **ATP** [[Paper](https://arxiv.org/abs/2206.01720)][[Code](https://github.com/StanfordVL/atp-video-language)][[Website](https://stanfordvl.github.io/atp-revisit-video-lang/)][[Poster](https://stanfordvl.github.io/atp-revisit-video-lang//assets/atp_poster_cvpr2022.pdf)][[Oral](https://youtu.be/-qVZKaP7iR0)] @Stanford
    <br/> Revisiting the "Video" in Video-Language Understanding (CVPR 2022)

  * **InternVideo** [[Paper](https://arxiv.org/abs/2212.03191)][[Code](https://github.com/OpenGVLab/InternVideo)] @OpenGVLab
    <br/> InternVideo: General Video Foundation Models via Generative and Discriminative Learning (arXiv 2022)

  * **VIOLET** [[Paper](https://arxiv.org/abs/2111.12681)][[Code](https://github.com/tsujuifu/pytorch_violet)] @Microsoft
    <br/> VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling (arXiv 2021)

  * **VidLanKD** [[Paper](https://arxiv.org/abs/2107.02681)][[Code](https://github.com/zinengtang/VidLanKD)] @UNC
    <br/> VidLanKD: Improving Language Understanding via Video-Distilled Knowledge Transfer  (NeurIPS 2021)

  * **MCN** [[Paper](https://arxiv.org/abs/2104.12671)][[Code](https://github.com/brian7685/Multimodal-Clustering-Network)] @MIT-IBM
    <br/> Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos (ICCV 2021)

  * **HERO** [[Paper](https://arxiv.org/abs/2005.00200)][[Code](https://github.com/linjieli222/HERO)] @Microsoft
    <br/> HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training (EMNLP 2020)

  * **UniVL** [[Paper](https://arxiv.org/abs/2002.06353)][[Code](https://github.com/microsoft/UniVL)] @Microsoft
    <br/> UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation (arXiv 2020)


## Video Retrieval

  * **FiT** [[Paper](https://arxiv.org/abs/2104.00650)][[Code](https://github.com/m-bain/frozen-in-time)][[Website](https://www.robots.ox.ac.uk/~vgg/research/frozen-in-time)][[Demo](http://meru.robots.ox.ac.uk/frozen-in-time)] @Oxford
    <br/> Frozen in Time: ️A Joint Video and Image Encoder for End to End Retrieval (ICCV 2021)

  * **CLIP-Hitchhiker** [[Paper](https://arxiv.org/abs/2205.08508)] @Oxford
    <br/> A CLIP-Hitchhiker's Guide to Long Video Retrieval (arXiv 2022)

  * **CLIP2Video** [[Paper](https://arxiv.org/abs/2106.11097)][[Code](https://github.com/CryhanFang/CLIP2Video)] @Tencent
    <br/> CLIP2Video: Mastering Video-Text Retrieval via Image CLIP (arXiv 2021)


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

  * **HMN** [[Paper](https://arxiv.org/abs/2111.12476)][[Code](https://github.com/MarcusNerva/HMN)] @UCAS
    <br/> Hierarchical Modular Network for Video Captioning (CVPR 2022)

  * **VX2TEXT** [[Paper](https://arxiv.org/abs/2101.12059)] @Facebook
    <br/> VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs (CVPR 2021)

  * **DeCEMBERT** [[Paper](https://aclanthology.org/2021.naacl-main.193)][[Code](https://github.com/zinengtang/DeCEMBERT)][[Oral](https://underline.io/events/122/sessions/4187/lecture/20006-decembert-learning-from-noisy-instructional-videos-via-dense-captions-and-entropy-minimization)] @UNC
    <br/> DeCEMBERT: Learning from Noisy Instructional Videos via Dense Captions and Entropy Minimization (NAACL 2021)

  * **CLIP4Caption** [[Paper](https://arxiv.org/abs/2110.06615)] @Tencent
    <br/> CLIP4Caption: CLIP for Video Caption (ACM 2021)

  * **ViTT** [[Paper](https://arxiv.org/abs/2011.11760)][[Oral](https://youtu.be/lahYNDzMyUs)] @Google
    <br/> Multimodal Pretraining for Dense Video Captioning (ACL 2020)


## Others

  * **GPT2MVS** [[Paper](https://arxiv.org/abs/2104.12465)] @UvA
    <br/> GPT2MVS: Generative Pre-trained Transformer-2 for Multi-modal Video Summarization (ICMR 2021)

# Datasets and SOTA

## Large-scale Video Language Dataset

  * **WebVid-10M** [[Paper](https://arxiv.org/abs/2104.00650)][[Code](https://github.com/m-bain/webvid)][[Website](https://m-bain.github.io/webvid-dataset)] @Oxford
    <br/> Frozen in Time: A Joint Video and Image Encoder for End to End Retrieval (ICCV 2021)

  * **HowTo100M** [[Paper](https://arxiv.org/abs/1906.03327)][[Code](https://github.com/antoine77340/howto100m)][[Website](https://www.di.ens.fr/willow/research/howto100m)] @Inria
    <br/> HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips (ICCV 2019)

## Downstream Tasks

  * **MAD-v2** [[Paper](https://arxiv.org/abs/2303.16899)][[Code](https://github.com/TengdaHan/AutoAD)][[Website](https://www.robots.ox.ac.uk/~vgg/research/autoad)] @Oxford
    <br/> AutoAD: Movie Description in Context (CVPR 2023)

  * **AGQA-Decomp** [[Paper](https://arxiv.org/abs/2204.07190)][[Code](https://github.com/madeleinegrunde/AGQA_baselines_code)][[Website](https://agqa-decomp.cs.washington.edu)] @UW
    <br/> Measuring Compositional Consistency for Video Question Answering (CVPR 2022)

  * **MAD** [[Paper](https://arxiv.org/abs/2112.00431)][[Code](https://github.com/Soldelli/MAD)][[PapersWithCode](https://paperswithcode.com/dataset/mad)] @KAUST
    <br/> MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions (CVPR 2022)

  * **QVHighlights** [[Paper](https://arxiv.org/abs/2107.09609)][[Code](https://github.com/jayleicn/moment_detr)][[PapersWithCode](https://paperswithcode.com/dataset/qvhighlights)] @UNC
    <br/> QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries (NeurIPS 2021)

  * **STAR** [[Paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/5ef059938ba799aaa845e1c2e8a762bd-Abstract-round2.html)][[Code](https://github.com/csbobby/STAR_Benchmark)][[Website](https://bobbywu.com/STAR)][[PaperswithCode](https://paperswithcode.com/dataset/star-1)] @MIT-IBM
    <br/> STAR: A Benchmark for Situated Reasoning in Real-World Videos (NeurIPS 2021)

  * **VidSitu** [[Paper](https://arxiv.org/abs/2104.00990)][[Code](https://github.com/TheShadow29/VidSitu)][[Website](https://vidsitu.org)][[PapersWithCode](https://paperswithcode.com/dataset/vidsitu)] @AI2
    <br/> Visual Semantic Role Labeling for Video Understanding (CVPR 2021)

  * **AGQA** [[Paper](https://arxiv.org/abs/2103.16002)][[Code](https://github.com/madeleinegrunde/AGQA_baselines_code)][[Website](https://cs.stanford.edu/people/ranjaykrishna/agqa)][[PapersWithCode](https://paperswithcode.com/dataset/agqa)] @Stanford
    <br/> AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning (CVPR 2021)

  * **LVU** [[Paper](https://arxiv.org/abs/2106.11310)][[Code](https://github.com/chaoyuaw/lvu)][[Website](https://chaoyuan.org/lvu)] @UTAUS
    <br/> Towards Long-Form Video Understanding (CVPR 2021)

  * **DramaQA** [[Paper](https://arxiv.org/abs/2005.03356)][[Code](https://github.com/liveseongho/DramaQA)][[Website](https://dramaqa.snu.ac.kr)][[PapersWithCode](https://paperswithcode.com/dataset/dramaqa)] @SNU
    <br/> DramaQA: Character-Centered Video Story Understanding with Hierarchical QA (AAAI 2021)

  * **How2QA** [[Paper](https://arxiv.org/abs/2005.00200)][[Code](https://github.com/linjieli222/HERO)] @Microsoft
    <br/> HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training (EMNLP 2020)

  * **How2R** [[Paper](https://arxiv.org/abs/2005.00200)][[Code](https://github.com/linjieli222/HERO)] @Microsoft
    <br/> HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training (EMNLP 2020)

  * **VLEP** [[Paper](https://arxiv.org/abs/2010.07999)][[Code](https://github.com/jayleicn/VideoLanguageFuturePred)][[PapersWithCode](https://paperswithcode.com/dataset/vlep)] @UNC
  <br/> What is More Likely to Happen Next? Video-and-Language Future Event Prediction (EMNLP 2020)

  * **V2C** [[Paper](https://arxiv.org/abs/2003.05162)][[Code](https://github.com/jacobswan1/Video2Commonsense)][[Website](https://asu-apg.github.io/Video2Commonsense)][[PapersWithCode](https://paperswithcode.com/paper/video2commonsense-generating-commonsense)] @ASU
    <br/> Video2Commonsense: Generating Commonsense Descriptions to Enrich Video Captioning (EMNLP 2020)

  * **CMD** [[Paper](https://arxiv.org/abs/2005.04208)][[Code](https://github.com/m-bain/CondensedMovies)][[Website](https://www.robots.ox.ac.uk/~vgg/research/condensed-movies)][[PapersWithCode](https://paperswithcode.com/dataset/cmd)] @Oxford
    <br/> Condensed Movies: Story Based Retrieval with Contextual Embeddings (ACCV 2020)

  * **TVC** [[Paper](https://arxiv.org/abs/2001.09099)][[Code](https://github.com/jayleicn/TVCaption)][[Website](https://tvr.cs.unc.edu/tvc.html)] @UNC
    <br/> TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval (ECCV 2020)

  * **TVR** [[Paper](https://arxiv.org/abs/2001.09099)][[Code](https://github.com/jayleicn/TVRetrieval)][[Website](https://tvr.cs.unc.edu)][[PapersWithCode](https://paperswithcode.com/dataset/tvr)] @UNC
    <br/> TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval (ECCV 2020)

  * **VIOLIN** [[Paper](https://arxiv.org/abs/2003.11618)][[Code](https://github.com/jimmy646/violin)][[PapersWithCode](https://paperswithcode.com/dataset/violin)] @Microsoft
    <br/> VIOLIN: A Large-Scale Dataset for Video-and-Language Inference (CVPR 2020)

  * **KnowITVQA** [[Paper](https://arxiv.org/abs/1910.10706)][[Code](https://github.com/noagarcia/knowit-rock)][[Website](https://knowit-vqa.github.io)][[PapersWithCode](https://paperswithcode.com/dataset/knowit-vqa)] @Osaka
    <br/> KnowIT VQA: Answering Knowledge-Based Questions about Videos (AAAI 2020)

  * **TVQA+** [[Paper](https://arxiv.org/abs/1904.11574)][[Code](https://github.com/jayleicn/TVQAplus)][[Website](https://tvqa.cs.unc.edu)][[PapersWithCode](https://paperswithcode.com/dataset/tvqa-1)] @UNC
    <br/> TVQA+: Spatio-Temporal Grounding for Video Question Answering (ACL 2020)
  
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

  * **TGIF-QA** [[Paper](https://arxiv.org/abs/1704.04497)/[Journal](https://link.springer.com/article/10.1007/s11263-019-01189-x)][[Code](https://github.com/YunseokJANG/tgif-qa)][[PapersWithCode](https://paperswithcode.com/dataset/tgif-qa)] @SNU
    <br/> TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering (CVPR 2016)
    <br/> Video Question Answering with Spatio-Temporal Reasoning (IJCV 2019)

  * **PororoQA** [[Paper](https://arxiv.org/abs/1707.00836)][[Code](https://github.com/Kyung-Min/Deep-Embedded-Memory-Networks)] @SNU
    <br/> DeepStory: Video Story QA by Deep Embedded Memory Networks (IJCAI 2017)

  * **LSMDC** [[Paper](https://arxiv.org/abs/1605.03705)][[Website](https://sites.google.com/site/describingmovies)][[PapersWithCode](https://paperswithcode.com/dataset/lsmdc)] @MPII
    <br/> Movie Description (IJCV 2017)

  * **MovieQA** [[Paper](https://arxiv.org/abs/1512.02902)][[Code](https://github.com/makarandtapaswi/MovieQA_benchmark)][[PapersWithCode](https://paperswithcode.com/dataset/movieqa)] @UToronto
    <br/> MovieQA: Understanding Stories in Movies through Question-Answering (CVPR 2016)

  * **MSR-VTT** [[Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf)][[PapersWithCode](https://paperswithcode.com/dataset/msr-vtt)] @Microsoft
    <br/> MSR-VTT: A Large Video Description Dataset for Bridging Video and Language (CVPR 2016)

  * **MPII-MD** [[Paper](https://arxiv.org/abs/1501.02530)][[Website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset)][[PapersWithCode](https://paperswithcode.com/dataset/lsmdc)] @MPII
    <br/> A Dataset for Movie Description (CVPR 2015)





