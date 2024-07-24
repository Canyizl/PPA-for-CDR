# [CIKM 2024] Preference Prototype-Aware Learning for Universal Cross-Domain Recommendation



Official codebase for the paper Preference Prototype-Aware Learning for Universal Cross-Domain Recommendation.



## Overview

![overview](https://github.com/Canyizl/PPA-for-CDR/blob/main/fig/introfig.png)

**Abstract:** Cross-domain recommendation~(CDR) aims to suggest items from new domains that align with potential user preferences, based on their historical interactions. Existing methods primarily focus on acquiring item representations by discovering user preferences under specific, yet possibly redundant, item features. However, user preferences may be more strongly associated with interacted items at higher semantic levels, rather than specific item features. Consequently, this item feature-focused recommendation approach can easily become suboptimal or even obsolete when conducting CDR with disturbances of these redundant features. In this paper, we propose a novel Preference Prototype-Aware~(PPA) learning method to quantitatively learn user preferences while minimizing disturbances from the source domain. The PPA framework consists of two complementary components: a mix-encoder and a preference prototype-aware decoder, forming an end-to-end unified framework suitable for various real-world scenarios. The mix-encoder employs a mix-network to learn better general representations of interacted items and capture the intrinsic relationships between items across different domains. The preference prototype-aware decoder implements a learnable prototype matching mechanism to quantitatively perceive user preferences, which can accurately capture user preferences at a higher semantic level. This decoder can also avoid disturbances caused by item features from the source domain. The experimental results on public benchmark datasets in different scenarios demonstrate the superiority of the proposed PPA learning method compared to state-of-the-art counterparts. PPA excels not only in providing accurate recommendations but also in offering reliable preference prototypes.



## Datasets

We use the datasets provided by [UniCDR](https://github.com/cjx96/UniCDR)

All used datasets can be downloaded at [WSDM2023-UniCDR-datasets](https://drive.google.com/drive/folders/1DCYiFU6GCVj681GKYUY2d_BJFln1-8gL?usp=share_link) (including 4 CDR scenarios: dual-user-intra, dual-user-inter, multi-item-intra, and multi-user-intra).

Note that all datasets are required to unzip in the root directory.



## Usage

Running example:

```shell
# dual-user-intra
CUDA_VISIBLE_DEVICES=0 python -u train_rec.py  --static_sample --cuda --domains sport_cloth --aggregator Transformer > Tppa_dual_user_intra_sport_cloth.log 2>&1&

# dual-user-inter
CUDA_VISIBLE_DEVICES=0 python -u train_rec.py  --static_sample --cuda --domains game_video --task dual-user-inter --aggregator Transformer > Tppa_dual_item_inter_game_video.log 2>&1&


# multi-item-intra
CUDA_VISIBLE_DEVICES=1   python -u train_rec.py  --static_sample --cuda --domains m1_m2_m3_m4_m5 --task multi-item-intra --aggregator Transformer > Tppa_multi_item_intra.log 2>&1&


# multi-user-intra
CUDA_VISIBLE_DEVICES=1 python -u train_rec.py --static_sample --cuda --domains d1_d2_d3 --task multi-user-intra --aggregator Transformer > Tppa_multi_user_intra.log 2>&1&
```
