
# Deep Learning for 3D Human Pose Estimation and Mesh Recovery: A Survey

Authors: Yang Liu, Changzhen Qiu, Zhiyong Zhang*

School of Electronics and Communication Engineering, Sun Yat-sen University, Shenzhen, Guangdong, China

### Overview
This is the regularly updated project page of Deep Learning for 3D Human Pose Estimation and Mesh Recovery: A Survey, a review that primarily concentrates on deep learning approaches to 3D human pose estimation and human mesh recovery. This survey comprehensively includes the most recent state-of-the-art publications (2019-now) from mainstream computer vision conferences and journals.

Please create issues if you have any suggestions!

## Citation
### Please kindly cite the papers if our work is useful and helpful for your research.

    @article{liu2024deep,
          title={Deep learning for 3D human pose estimation and mesh recovery: A survey}, 
          author={Liu, Yang and Qiu, Changzhen and Zhang, Zhiyong},
          journal={Neurocomputing},
          pages={128049},
          year={2024},
          issn={0925-2312},
          doi={https://doi.org/10.1016/j.neucom.2024.128049},
          publisher={Elsevier}
    }

## 3D human pose estimation
- Single Person
    - In Images
        - Solving Depth Ambiguity
            - Optical-aware: VI-HC [[paper]](https://ieeexplore.ieee.org/abstract/document/8763991/), Ray3D [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Zhan_Ray3D_Ray-Based_3D_Human_Pose_Estimation_for_Monocular_Absolute_3D_CVPR_2022_paper.html)
            - Appropriate feature representation: HEMlets [[paper]](https://ieeexplore.ieee.org/abstract/document/9320561/)
            - Joint aware: JRAN [[paper]](https://ieeexplore.ieee.org/abstract/document/8995784/)
        - Solving Body Structure Understanding
            - Limb aware: Wu et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9663053/), Deep grammar network [[paper]](https://ieeexplore.ieee.org/abstract/document/9450016/)
            - Orientation keypoints: Fisch et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9653865/)
            - Graph-based: Liu et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/8621059/), LCN [[paper]](https://ieeexplore.ieee.org/abstract/document/9174911/), Modulated-CNN [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Zou_Modulated_Graph_Convolutional_Network_for_3D_Human_Pose_Estimation_ICCV_2021_paper.html), Skeletal-GNN [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Zeng_Learning_Skeletal_Graph_Neural_Networks_for_Hard_3D_Pose_Estimation_ICCV_2021_paper.html), HopFIR [[paper]](https://arxiv.org/abs/2302.14581), RS-Net [[paper]](https://ieeexplore.ieee.org/abstract/document/10179252/)
        - Solving Occlusion Problems
            - Learnable-triangulation [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Iskakov_Learnable_Triangulation_of_Human_Pose_ICCV_2019_paper.html)
            - RPMS [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Qiu_Cross_View_Fusion_for_3D_Human_Pose_Estimation_ICCV_2019_paper.html)
            - Lightweight multi-view [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Remelli_Lightweight_Multi-View_3D_Pose_Estimation_Through_Camera-Disentangled_Representation_CVPR_2020_paper.html)
            - AdaFuse [[paper]](https://link.springer.com/article/10.1007/s11263-020-01398-9)
            - Bartol et al. [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Bartol_Generalizable_Human_Pose_Triangulation_CVPR_2022_paper.html)
            - 3D pose consensus [[paper]](https://link.springer.com/article/10.1007/s11263-021-01570-9)
            - Probabilistic triangulation model [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Jiang_Probabilistic_Triangulation_for_Uncalibrated_Multi-View_3D_Human_Pose_Estimation_ICCV_2023_paper.html)
        - Solving Data Lacking
            - Unsupervised learning: Kudo et al. [[paper]](https://arxiv.org/abs/1803.08244), Chen et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Unsupervised_3D_Pose_Estimation_With_Geometric_Self-Supervision_CVPR_2019_paper.html), ElePose [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Wandt_ElePose_Unsupervised_3D_Human_Pose_Estimation_by_Predicting_Camera_Elevation_CVPR_2022_paper.html)
            - Self-supervised learning: EpipolarPose [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Kocabas_Self-Supervised_Learning_of_3D_Human_Pose_Using_Multi-View_Geometry_CVPR_2019_paper.html), Wang et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/8611195/), MRP-Net [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Kundu_Uncertainty-Aware_Adaptation_for_Self-Supervised_3D_Human_Pose_Estimation_CVPR_2022_paper.html), PoseTriplet [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Gong_PoseTriplet_Co-Evolving_3D_Human_Pose_Estimation_Imitation_and_Hallucination_Under_CVPR_2022_paper.html)
            - Weakly-supervised learning: Hua et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9765377/), CameraPose [[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Yang_CameraPose_Weakly-Supervised_Monocular_3D_Human_Pose_Estimation_by_Leveraging_In-the-Wild_WACV_2023_paper.html)
            - Transfer learning: Adaptpose [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Gholami_AdaptPose_Cross-Dataset_Adaptation_for_3D_Human_Pose_Estimation_by_Learnable_CVPR_2022_paper.html)
    - In Videos
        - Solving Single-frame Limitation
            - VideoPose3D [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html)
            - PoseFormer [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Zheng_3D_Human_Pose_Estimation_With_Spatial_and_Temporal_Transformers_ICCV_2021_paper.html)
            - UniPose+ [[paper]](https://ieeexplore.ieee.org/abstract/document/9599531/)
            - MHFormer [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Li_MHFormer_Multi-Hypothesis_Transformer_for_3D_Human_Pose_Estimation_CVPR_2022_paper.html)
            - MixSTE [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MixSTE_Seq2seq_Mixed_Spatio-Temporal_Encoder_for_3D_Human_Pose_Estimation_CVPR_2022_paper.html)
            - Honari et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9921314/)
            - HSTFormer [[paper]](https://arxiv.org/abs/2301.07322)
            - STCFormer [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Tang_3D_Human_Pose_Estimation_With_Spatio-Temporal_Criss-Cross_Attention_CVPR_2023_paper.html)
        - Solving Real-time Problems
            - Temporally sparse sampling: Einfalt et al. [[paper]](https://openaccess.thecvf.com/content/WACV2023/html/Einfalt_Uplift_and_Upsample_Efficient_3D_Human_Pose_Estimation_With_Uplifting_WACV_2023_paper.html)
            - Spatio-temporal sparse sampling: MixSynthFormer [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Sun_MixSynthFormer_A_Transformer_Encoder-like_Structure_with_Mixed_Synthetic_Self-attention_for_ICCV_2023_paper.html)
        - Solving Body Structure Understanding
            - Motion loss: Wang et al. [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_45)
            - Human-joint affinity: Dc-Net [[paper]](https://ieeexplore.ieee.org/abstract/document/9531423/)
            - Anatomy-aware: Chen et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9347537/)
            - Part aware attention: Xue et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9798770/)
        - Solving Occlusion Problems
            - Optical-flow consistency constraint: Cheng et al. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Cheng_Occlusion-Aware_Networks_for_3D_Human_Pose_Estimation_in_Video_ICCV_2019_paper.html)
            - Multi-view: MTF-Transformer [[paper]](https://ieeexplore.ieee.org/abstract/document/9815549/)
        - Solving Data Lacking
            - Unsupervised learning: Yu et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Yu_Towards_Alleviating_the_Modeling_Ambiguity_of_Unsupervised_Monocular_3D_Human_ICCV_2021_paper.html)
            - Weakly-supervised learning: Chen et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.html)
            - Semi-supervised learning: MCSS [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Mitra_Multiview-Consistent_Semi-Supervised_Learning_for_3D_Human_Pose_Estimation_CVPR_2020_paper.html)
            - Self-supervised learning: Kundu et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Kundu_Self-Supervised_3D_Human_Pose_Estimation_via_Part_Guided_Novel_Image_CVPR_2020_paper.html), P-STMO [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_27)
            - Meta-learning: Cho et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Cho_Camera_Distortion-Aware_3D_Human_Pose_Estimation_in_Video_With_Optimization-Based_ICCV_2021_paper.html)
            - Data augmentation: PoseAug [[paper]](http://openaccess.thecvf.com/content/CVPR2021/html/Gong_PoseAug_A_Differentiable_Pose_Augmentation_Framework_for_3D_Human_Pose_CVPR_2021_paper.html), Zhang et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/10050391/)
- Multi-person
    - Top-down
        - Solving Real-time Problems
            - Multi-view: Chen et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Cross-View_Tracking_for_Multi-Human_3D_Pose_Estimation_at_Over_100_CVPR_2020_paper.html)
            - Whole body: AlphaPose [[paper]](https://ieeexplore.ieee.org/abstract/document/9954214/)
        - Solving Representation Limitation
            - VoxelTrack [[paper]](https://ieeexplore.ieee.org/abstract/document/9758679/)
        - Solving Occlusion Problems
            - Wu et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Wu_Graph-Based_3D_Multi-Person_Pose_Estimation_Using_Multi-View_Images_ICCV_2021_paper.html)
        - Solving Data Lacking
            - Single-shot: PandaNet [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Benzine_PandaNet_Anchor-Based_Single-Shot_Multi-Person_3D_Pose_Estimation_CVPR_2020_paper.html)
            - Optical-aware: Moon et al. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Moon_Camera_Distance-Aware_Top-Down_Approach_for_3D_Multi-Person_Pose_Estimation_From_ICCV_2019_paper.html)
    - Bottom-up
        -Solving Real-time Problems
            - Fabbri et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Fabbri_Compressed_Volumetric_Heatmaps_for_Multi-Person_3D_Pose_Estimation_CVPR_2020_paper.html)
        - Solving Supervisory Limitation. 
            - HMOR [[paper]](https://dl.acm.org/doi/abs/10.1007/978-3-030-58580-8_15)
        - Solving Data Lacking
            - Single-shot: SMAP [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_33), Benzine et al. [[paper]](https://www.sciencedirect.com/science/article/pii/S003132032030337X)
        - Solving Occlusion Problems
            - Mehta et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/8490962/)
            - LCR-Net++ [[paper]](https://ieeexplore.ieee.org/abstract/document/8611390/)
    - Others
        - Single Stage
            - Jin et al. [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Jin_Single-Stage_Is_Enough_Multi-Person_Absolute_3D_Pose_Estimation_CVPR_2022_paper.html)
        - Top-down & Bottom-up
            - Cheng et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9763389/)

## Human Mesh Recovery
- Template-based
    - Naked
        - Multimodal Methods
            - Hybrid annotations: Rong et al. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Rong_Delving_Deep_Into_Hybrid_Annotations_for_3D_Human_Recovery_in_ICCV_2019_paper.html)
            - Optical flow: DTS-VIBE [[paper]](https://openaccess.thecvf.com/content/WACV2022/html/Li_Deep_Two-Stream_Video_Inference_for_Human_Body_Pose_and_Shape_WACV_2022_paper.html)
            - Silhouettes: LASOR [[paper]](https://ieeexplore.ieee.org/abstract/document/9709705/)
            - Cropped image and bounding box: CLIFF [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_34)
        - Utilizing Attention Mechanism
            - Part-driven attention: PARE [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Kocabas_PARE_Part_Attention_Regressor_for_3D_Human_Body_Estimation_ICCV_2021_paper.html)
            - Graph attention: Mesh Graphormer [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Lin_Mesh_Graphormer_ICCV_2021_paper.html)
            - Spatio-temporal attention: MPS-Net [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Wei_Capturing_Humans_in_Motion_Temporal-Attentive_3D_Human_Pose_and_Shape_CVPR_2022_paper.html), PSVT [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Qiu_PSVT_End-to-End_Multi-Person_3D_Pose_and_Shape_Estimation_With_Progressive_CVPR_2023_paper.html)
            - Efficient architecture: FastMETRO [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_20), Xue et al. [[paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3548133)
            - End-to-end structure: METRO [[paper]](http://openaccess.thecvf.com/content/CVPR2021/html/Lin_End-to-End_Human_Pose_and_Mesh_Reconstruction_with_Transformers_CVPR_2021_paper.html)
        - Exploiting Temporal Information
            - Temporally encoding features: Kanazawa et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Kanazawa_Learning_3D_Human_Dynamics_From_Video_CVPR_2019_paper.html)
            - Self-attention temporal: VIBE [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.html)
            - Temporally consistent: TCMR [[paper]](http://openaccess.thecvf.com/content/CVPR2021/html/Choi_Beyond_Static_Features_for_Temporally_Consistent_3D_Human_Pose_and_CVPR_2021_paper.html)
            - Multi-level spatial-temporal attention: MAED [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Wan_Encoder-Decoder_With_Multi-Level_Attention_for_3D_Human_Shape_and_Pose_ICCV_2021_paper.html)
            - Temporally embedded live stream: TePose [[paper]](https://arxiv.org/abs/2207.12537)
            - Short-term and long-term temporal correlations: Glot [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Shen_Global-to-Local_Modeling_for_Video-Based_3D_Human_Pose_and_Shape_Estimation_CVPR_2023_paper.html)
        - Multi-view Methods
            - Confidence-aware majority voting mechanism: Dong et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Dong_Shape-Aware_Multi-Person_Pose_Estimation_From_Multi-View_Images_ICCV_2021_paper.html)
            - Probabilistic-based multi-view: Sengupta et al. [[paper]](http://openaccess.thecvf.com/content/CVPR2021/html/Sengupta_Probabilistic_3D_Human_Shape_and_Pose_Estimation_From_Multiple_Unconstrained_CVPR_2021_paper.html)
            - Dynamic physics-geometry consistency: Huang et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9665884/)
            - Cross-view fusion: Zhuo et al. [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Zhuo_Towards_Stable_Human_Pose_Estimation_via_Cross-View_Fusion_and_Foot_CVPR_2023_paper.html)
        - Boosting Efficiency
            - Sparse constrained formulation: SCOPE [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Fan_Revitalizing_Optimization_for_3D_Human_Pose_and_Shape_Estimation_A_ICCV_2021_paper.html)
            - Single-stage model: BMP [[paper]](http://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Body_Meshes_as_Points_CVPR_2021_paper.html)
            - Process heatmap inputs: HeatER [[paper]](https://arxiv.org/abs/2205.15448)
            - Removing redundant tokens: TORE [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Dou_TORE_Token_Reduction_for_Efficient_Human_Mesh_Recovery_with_Transformer_ICCV_2023_paper.html)
        - Developing Various Representations
            - Texture map: TexturePose [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Pavlakos_TexturePose_Supervising_Human_Mesh_Estimation_With_Texture_Consistency_ICCV_2019_paper.html)
            - UV map: Zhang et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9279291/), DecoMR [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Zeng_3D_Human_Mesh_Regression_With_Dense_Correspondence_CVPR_2020_paper.html), Zhang et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Object-Occluded_Human_Shape_and_Pose_Estimation_From_a_Single_Color_CVPR_2020_paper.html)
            - Heat map: Sun et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Sun_Monocular_One-Stage_Regression_of_Multiple_3D_People_ICCV_2021_paper.html), 3DCrowdNet [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Choi_Learning_To_Estimate_Robust_3D_Human_Mesh_From_In-the-Wild_Crowded_CVPR_2022_paper.html)
            - Uniform representation: DSTFormer [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html)
        - Utilizing Structural Information
            - Part-based: holopose [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Guler_HoloPose_Holistic_3D_Human_Reconstruction_In-The-Wild_CVPR_2019_paper.html)
            - Skeleton disentangling: Sun et al. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Sun_Human_Mesh_Recovery_From_Monocular_Images_via_a_Skeleton-Disentangled_Representation_ICCV_2019_paper.html)
            - Hybrid inverse kinematics: HybrIK [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Li_HybrIK_A_Hybrid_Analytical-Neural_Inverse_Kinematics_Solution_for_3D_Human_CVPR_2021_paper.html?ref=https://githubhelp.com), NIKI [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Li_NIKI_Neural_Inverse_Kinematics_With_Invertible_Neural_Networks_for_3D_CVPR_2023_paper.html)
            - Uncertainty-aware: Lee et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Lee_Uncertainty-Aware_Human_Mesh_Recovery_From_Video_by_Learning_Part-Based_3D_ICCV_2021_paper.html)
            - Kinematic tree structure: Sengupta et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Sengupta_Hierarchical_Kinematic_Probability_Distributions_for_3D_Human_Shape_and_Pose_ICCV_2021_paper.html)
            - Kinematic chains: SGRE [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Wang_3D_Human_Mesh_Recovery_with_Sequentially_Global_Rotation_Estimation_ICCV_2023_paper.html)
        - Choosing Appropriate Learning Strategies
            - Self-improving: SPIN [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.html), ReFit [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Wang_ReFit_Recurrent_Fitting_Network_for_3D_Human_Recovery_ICCV_2023_paper.html), You et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/You_Co-Evolution_of_Pose_and_Mesh_for_3D_Human_Body_Estimation_ICCV_2023_paper.html)
            - Novel losses: Zanfir et al. [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_28), Jiang et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_Coherent_Reconstruction_of_Multiple_Humans_From_a_Single_Image_CVPR_2020_paper.html)
            - Unsupervised learning: Madadi et al. [[paper]](https://link.springer.com/article/10.1007/s11263-021-01488-2), Yu et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Yu_Skeleton2Mesh_Kinematics_Prior_Injected_Unsupervised_Human_Mesh_Recovery_ICCV_2021_paper.html)
            - Bilevel online adaptation: Guan et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9842366/)
            - Single-shot: Pose2UV [[paper]](https://ieeexplore.ieee.org/abstract/document/9817035/)
            - Contrastive learning: JOTR [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Li_JOTR_3D_Joint_Contrastive_Learning_with_Transformers_for_Occluded_Human_ICCV_2023_paper.html)
            - Domain adaptation: Nam et al. [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Nam_Cyclic_Test-Time_Adaptation_on_Monocular_Video_for_3D_Human_Mesh_ICCV_2023_paper.html)
    - Detailed
        - With Clothes
            - Alldieck et al. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Alldieck_Learning_to_Reconstruct_People_in_Clothing_From_a_Single_RGB_CVPR_2019_paper.html)
            - Multi-Garment Network (MGN) [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Bhatnagar_Multi-Garment_Net_Learning_to_Dress_3D_People_From_Images_ICCV_2019_paper.html)
            - Texture map: Tex2Shape [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Alldieck_Tex2Shape_Detailed_Full_Human_Body_Geometry_From_a_Single_Image_ICCV_2019_paper.html)
            - Layered garment representation: BCNet [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_2)
            - Temporal span: H4D [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Jiang_H4D_Human_4D_Modeling_by_Learning_Neural_Compositional_Representation_CVPR_2022_paper.html)
        - With Hands
            - Linguistic priors: SGNify [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html)
            - Two-hands interaction: [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Interacting_Two-Hand_3D_Pose_and_Shape_Reconstruction_From_Single_Color_ICCV_2021_paper.html)
            - Hand-object interaction: [[paper]](https://ieeexplore.ieee.org/abstract/document/9390307/)
        - Whole Body
            - PROX [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Hassan_Resolving_3D_Human_Pose_Ambiguities_With_3D_Scene_Constraints_ICCV_2019_paper.html)
            - ExPose [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_2)
            - FrankMocap [[paper]](https://openaccess.thecvf.com/content/ICCV2021W/ACVR/html/Rong_FrankMocap_A_Monocular_3D_Whole-Body_Pose_Estimation_System_via_Regression_ICCVW_2021_paper.html)
            - PIXIE [[paper]](https://ieeexplore.ieee.org/abstract/document/9665886/)
            - Moon et al. [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/ABAW/html/Moon_Accurate_3D_Hand_Pose_Estimation_for_Whole-Body_3D_Human_Mesh_CVPRW_2022_paper.html)
            - PyMAF [[paper]](https://ieeexplore.ieee.org/abstract/document/10113183/)
            - OSX [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.html)
            - HybrIK-X [[paper]](https://arxiv.org/abs/2304.05690)
- Template-free
    - Regression-based
        - FACSIMILE [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Smith_FACSIMILE_Fast_and_Accurate_Scans_From_an_Image_in_Less_ICCV_2019_paper.html), PeeledHuman [[paper]](https://ieeexplore.ieee.org/abstract/document/9320367/), GTA [[paper]](https://arxiv.org/abs/2309.13524), NSF [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Xue_NSF_Neural_Surface_Fields_for_Human_Modeling_from_Monocular_Depth_ICCV_2023_paper.html)
    - Optimization-based Differentiable
        - DiffPhy [[paper]](http://openaccess.thecvf.com/content/CVPR2022/html/Gartner_Differentiable_Dynamics_for_Articulated_3D_Human_Motion_Reconstruction_CVPR_2022_paper.html), AG3D [[paper]](https://arxiv.org/abs/2305.02312)
    - Implicit Representations
        - PIFu [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/html/Saito_PIFu_Pixel-Aligned_Implicit_Function_for_High-Resolution_Clothed_Human_Digitization_ICCV_2019_paper.html), PIFuHD [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Saito_PIFuHD_Multi-Level_Pixel-Aligned_Implicit_Function_for_High-Resolution_3D_Human_Digitization_CVPR_2020_paper.html)
        - Canonical space: ARCH [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_ARCH_Animatable_Reconstruction_of_Clothed_Humans_CVPR_2020_paper.html), ARCH++ [[paper]](http://openaccess.thecvf.com/content/ICCV2021/html/He_ARCH_Animation-Ready_Clothed_Human_Reconstruction_Revisited_ICCV_2021_paper.html), CAR [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Liao_High-Fidelity_Clothed_Avatar_Reconstruction_From_a_Single_Image_CVPR_2023_paper.html)
        - Geometric priors: GeoPIFu [[paper]](https://proceedings.neurips.cc/paper/2020/hash/690f44c8c2b7ded579d01abe8fdb6110-Abstract.html)
        - Novel representations: Peng et al. [[paper]](http://openaccess.thecvf.com/content/CVPR2021/html/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.html), 3DNBF [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Zhang_3D-Aware_Neural_Body_Fitting_for_Occlusion_Robust_3D_Human_Pose_ICCV_2023_paper.html)
    - Neural Radiance Fields
        - Volume deformation scheme [[paper]](https://ieeexplore.ieee.org/abstract/document/9888037/)
        - ActorsNeRF [[paper]](https://arxiv.org/abs/2304.14401)
    - Diffusion Models
        - HMDiff [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Foo_Distribution-Aligned_Diffusion_for_Human_Mesh_Recovery_ICCV_2023_paper.html)
    - Implicit + Explicit
        - HMD [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Detailed_Human_Shape_Estimation_From_a_Single_Image_by_Hierarchical_CVPR_2019_paper.html), IP-Net [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_19), PaMIR [[paper]](https://ieeexplore.ieee.org/abstract/document/9321139/), Zhu et al. [[paper]](https://ieeexplore.ieee.org/abstract/document/9507281/), ICON [[paper]](https://ieeexplore.ieee.org/abstract/document/9878790/), ECON [[paper]](http://openaccess.thecvf.com/content/CVPR2023/html/Xiu_ECON_Explicit_Clothed_Humans_Optimized_via_Normal_Integration_CVPR_2023_paper.html), DELTA [[paper]](https://arxiv.org/abs/2309.06441), GETAvatar [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Zhang_GETAvatar_Generative_Textured_Meshes_for_Animatable_Human_Avatars_ICCV_2023_paper.html)
    - Diffusion + Explicit
        - DINAR [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Svitov_DINAR_Diffusion_Inpainting_of_Neural_Textures_for_One-Shot_Human_Avatars_ICCV_2023_paper.html)
    - NeRF + Explicit
        - TransHuman [[paper]](http://openaccess.thecvf.com/content/ICCV2023/html/Pan_TransHuman_A_Transformer-based_Human_Representation_for_Generalizable_Neural_Human_Rendering_ICCV_2023_paper.html)
    - Gaussian Splatting + Explicit
        - Animatable 3D Gaussian [[paper]](https://arxiv.org/abs/2311.16482)

## The overview of the mainstream datasets.
| Dataset             | Type       | Data  | Total frames | Feature                     | Download link                                      |
|---------------------|------------|-------|--------------|-----------------------------|----------------------------------------------------|
| Human3.6M           | 3D/Mesh    | Video | 3.6M         | multi-view                  | [Website](http://vision.imar.ro/human3.6m/description.php) |
| 3DPW                | 3D/Mesh    | Video | 51K          | multi-person                | [Website](https://virtualhumans.mpi-inf.mpg.de/3DPW/) |
| MPI-INF-3DHP        | 2D/3D      | Video | 2K           | in-wild                     | [Website](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) |
| HumanEva            | 3D         | Video | 40K          | multi-view                  | [Website](http://humaneva.is.tue.mpg.de/) |
| CMU-Panoptic        | 3D         | Video | 1.5M         | multi-view/multi-person     | [Website](https://domedb.perception.cs.cmu.edu/) |
| MuCo-3DHP           | 3D         | Image | 8K           | multi-person/occluded scene | [Website](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) |
| SURREAL             | 2D/3D/Mesh | Video | 6.0M         | synthetic model             | [Website](https://www.di.ens.fr/willow/research/surreal/data/) |
| 3DOH50K             | 2D/3D/Mesh | Image | 51K          | object-occluded             | [Website](https://www.yangangwang.com/#me) |
| 3DCP                | Mesh       | Mesh  | 190          | contact                     | [Website](https://tuch.is.tue.mpg.de/) |
| AMASS               | Mesh       | Motion| 11K          | soft-tissue dynamics        | [Website](https://amass.is.tue.mpg.de/) |
| DensePose           | Mesh       | Image | 50K          | multi-person                | [Website](http://densepose.org/) |
| UP-3D               | 3D/Mesh    | Image | 8K           | sport scene                 | [Website](https://files.is.tuebingen.mpg.de/classner/up/) |
| THuman2.0           | Mesh       | Image | 7K           | textured surface            | [Website](https://github.com/ytrock/THuman2.0-Dataset) |

## Comparisons of 3D pose estimation methods on Human3.6M.
| Method                | Year | Publication | Highlight                         | MPJPE↓ | PMPJPE↓ | Code                                                                                     |
|-----------------------|------|-------------|-----------------------------------|--------|---------|------------------------------------------------------------------------------------------|
| [Graformer](http://openaccess.thecvf.com/content/CVPR2022/html/Zhao_GraFormer_Graph-Oriented_Transformer_for_3D_Pose_Estimation_CVPR_2022_paper.html)    | 2022 | CVPR'22     | graph-based transformer           | 35.2   | -       | [Code](https://github.com/Graformer/GraFormer)                                             |
| [GLA-GCN](http://openaccess.thecvf.com/content/ICCV2023/html/Yu_GLA-GCN_Global-local_Adaptive_Graph_Convolutional_Network_for_3D_Human_Pose_ICCV_2023_paper.html)        | 2023 | ICCV'23     | adaptive GCN                      | 34.4   | 37.8    | [Code](https://github.com/bruceyo/GLA-GCN)                                               |
| [PoseDA](https://arxiv.org/abs/2303.16456)       | 2023 | arXiv'23    | domain adaptation                 | 49.4   | 34.2    | [Code](https://github.com/rese1f/PoseDA)                                                   |
| [GFPose](http://openaccess.thecvf.com/content/CVPR2023/html/Ci_GFPose_Learning_3D_Human_Pose_Prior_With_Gradient_Fields_CVPR_2023_paper.html)         | 2023 | CVPR'23     | gradient fields                   | 35.6   | 30.5    | [Code](https://sites.google.com/view/gfpose/)                                            |
| [TP-LSTMs](https://ieeexplore.ieee.org/abstract/document/9749007/)      | 2022 | TPAMI'22    | pose similarity metric            | 40.5   | 31.8    | -                                                                                         |
| [FTCM](https://ieeexplore.ieee.org/abstract/document/10159259/)         | 2023 | TCSVT'23    | frequency-temporal collaborative  | 28.1   | -       | [Code](https://github.com/zhenhuat/FTCM)                                                   |
| [VideoPose3D](http://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html)   | 2019 | CVPR'19  | semi-supervised                   | 46.8   | 36.5    | [Code](https://github.com/facebookresearch/VideoPose3D)                                      |
| [PoseFormer](http://openaccess.thecvf.com/content/ICCV2021/html/Zheng_3D_Human_Pose_Estimation_With_Spatial_and_Temporal_Transformers_ICCV_2021_paper.html)   | 2021 | ICCV'21   | spatio-temporal transformer       | 44.3   | 34.6    | [Code](https://github.com/zczcwh/PoseFormer)                                                 |
| [STCFormer](http://openaccess.thecvf.com/content/CVPR2023/html/Tang_3D_Human_Pose_Estimation_With_Spatio-Temporal_Criss-Cross_Attention_CVPR_2023_paper.html)    | 2023 | CVPR'23    | spatio-temporal transformer       | 40.5   | 31.8    | [Code](https://github.com/zhenhuat/STCFormer)                                               |
| [3Dpose_ssl](https://ieeexplore.ieee.org/abstract/document/8611195/)   | 2020 | TPAMI'20   | self-supervised                   | 63.6   | 63.7    | [Code](https://github.com/chanyn/3Dpose_ssl)                                                |
| [MTF-Transformer](https://ieeexplore.ieee.org/abstract/document/9815549/)   | 2022 | TPAMI'22 | multi-view temporal fusion        | 26.2   | -       | [Code](https://github.com/lelexx/MTF-Transformer)                                        |
| [AdaptPose](http://openaccess.thecvf.com/content/CVPR2022/html/Gholami_AdaptPose_Cross-Dataset_Adaptation_for_3D_Human_Pose_Estimation_by_Learnable_CVPR_2022_paper.html)   | 2022 | CVPR'22  | cross datasets                     | 42.5   | 34.0    | [Code](https://github.com/mgholamikn/AdaptPose)                                               |
| [3D-HPE-PAA](https://ieeexplore.ieee.org/abstract/document/9798770/)    | 2022 | TIP'22     | part aware attention               | 43.1   | 33.7    | [Code](https://github.com/thuxyz19/3D-HPE-PAA)                                            |
| [DeciWatch](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_35)    | 2022 | ECCV'22    | efficient framework                | 52.8   | -       | [Code](https://github.com/cure-lab/DeciWatch)                                              |
| [Diffpose](http://openaccess.thecvf.com/content/CVPR2023/html/Gong_DiffPose_Toward_More_Reliable_3D_Pose_Estimation_CVPR_2023_paper.html)     | 2023 | CVPR'23    | pose refine                        | 36.9   | 28.7    | [Code](https://gongjia0208.github.io/Diffpose/)                                            |
| [Elepose](http://openaccess.thecvf.com/content/CVPR2022/html/Wandt_ElePose_Unsupervised_3D_Human_Pose_Estimation_by_Predicting_Camera_Elevation_CVPR_2022_paper.html)     | 2022 | CVPR'22    | unsupervised                       | -      | 36.7    | [Code](https://github.com/bastianwandt/ElePose)                                             |
| [Uplift and Upsample](https://openaccess.thecvf.com/content/WACV2023/html/Einfalt_Uplift_and_Upsample_Efficient_3D_Human_Pose_Estimation_With_Uplifting_WACV_2023_paper.html)   | 2023 | CVPR'23 | efficient transformers            | 48.1   | 37.6    | [Code](https://github.com/goldbricklemon/uplift-upsample-3dhpe)                       |
| [RS-Net](https://ieeexplore.ieee.org/abstract/document/10179252/)     | 2023 | TIP'23     | regular splitting graph network    | 48.6   | 38.9    | [Code](https://github.com/nies14/RS-Net)                                                     |
| [HSTFormer](https://arxiv.org/abs/2301.07322)    | 2023 | arXiv'23   | spatial-temporal transformers      | 42.7   | 33.7    | [Code](https://github.com/qianxiaoye825/HSTFormer)                                         |
| [PoseFormerV2](http://openaccess.thecvf.com/content/CVPR2023/html/Zhao_PoseFormerV2_Exploring_Frequency_Domain_for_Efficient_and_Robust_3D_Human_CVPR_2023_paper.html)   | 2023 | CVPR'23  | frequency domain                   | 45.2   | 35.6    | [Code](https://github.com/QitaoZhao/PoseFormerV2)                                          |
| [DiffPose](http://openaccess.thecvf.com/content/ICCV2023/html/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.html)   | 2023 | ICCV'23  | diffusion models                   | 42.9   | 30.8    | [Code](https://github.com/bastianwandt/DiffPose/)                                              |

## Comparisons of 3D pose estimation methods on MPI-INF-3DHP.
| **Method** | **Year** | **Publication** | **Highlight** | **MPJPE↓** | **PCK↑** | **AUC↑** | **Code** |
|------------|----------|-----------------|-----------------------|-----------|----------|---------|----------|
| [HSTFormer](https://arxiv.org/abs/2301.07322)   | 2023 | arXiv'23 | spatial-temporal transformers | 28.3 | 98.0 | 78.6 | [Code](https://github.com/qianxiaoye825/HSTFormer) |
| [PoseFormerV2](http://openaccess.thecvf.com/content/CVPR2023/html/Zhao_PoseFormerV2_Exploring_Frequency_Domain_for_Efficient_and_Robust_3D_Human_CVPR_2023_paper.html)   | 2023 | CVPR'23 | frequency domain | 27.8 | 97.9 | 78.8 | [Code](https://github.com/QitaoZhao/PoseFormerV2) |
| [Uplift and Upsample](https://openaccess.thecvf.com/content/WACV2023/html/Einfalt_Uplift_and_Upsample_Efficient_3D_Human_Pose_Estimation_With_Uplifting_WACV_2023_paper.html)   | 2023 | CVPR'23 | efficient transformers | 46.9 | 95.4 | 67.6 | [Code](https://github.com/goldbricklemon/uplift-upsample-3dhpe) |
| [RS-Net](https://ieeexplore.ieee.org/abstract/document/10179252/)   | 2023 | TIP'23 | regular splitting graph network | - | 85.6 | 53.2 | [Code](https://github.com/nies14/RS-Net) |
| [Diffpose](http://openaccess.thecvf.com/content/CVPR2023/html/Gong_DiffPose_Toward_More_Reliable_3D_Pose_Estimation_CVPR_2023_paper.html)   | 2023 | CVPR'23 | pose refine | 29.1 | 98.0 | 75.9 | [Code](https://gongjia0208.github.io/Diffpose/) |
| [FTCM](https://ieeexplore.ieee.org/abstract/document/10159259/)   | 2023 | TCSVT'23 | frequency-temporal collaborative | 31.2 | 97.9 | 79.8 | [Code](https://github.com/zhenhuat/FTCM) |
| [STCFormer](http://openaccess.thecvf.com/content/CVPR2023/html/Tang_3D_Human_Pose_Estimation_With_Spatio-Temporal_Criss-Cross_Attention_CVPR_2023_paper.html)   | 2023 | CVPR'23 | spatio-temporal transformer | 23.1 | 98.7 | 83.9 | [Code](https://github.com/zhenhuat/STCFormer) |
| [PoseDA](https://arxiv.org/abs/2303.16456)   | 2023 | arXiv'23 | domain adaptation | 61.3 | 92.0 | 62.5 | [Code](https://github.com/rese1f/PoseDA) |
| [TP-LSTMs](https://ieeexplore.ieee.org/abstract/document/9749007/)   | 2022 | TPAMI'22 | pose similarity metric | 48.8 | 82.6 | 81.3 | - |
| [AdaptPose](http://openaccess.thecvf.com/content/CVPR2022/html/Gholami_AdaptPose_Cross-Dataset_Adaptation_for_3D_Human_Pose_Estimation_by_Learnable_CVPR_2022_paper.html)   | 2022 | CVPR'22 | cross datasets | 77.2 | 88.4 | 54.2 | [Code](https://github.com/mgholamikn/AdaptPose) |
| [3D-HPE-PAA](https://ieeexplore.ieee.org/abstract/document/9798770/)   | 2022 | TIP'22 | part aware attention | 69.4 | 90.3 | 57.8 | [Code](https://github.com/thuxyz19/3D-HPE-PAA) |
| [Elepose](http://openaccess.thecvf.com/content/CVPR2022/html/Wandt_ElePose_Unsupervised_3D_Human_Pose_Estimation_by_Predicting_Camera_Elevation_CVPR_2022_paper.html)   | 2022 | CVPR'22 | unsupervised | 54.0 | 86.0 | 50.1 | [Code](https://github.com/bastianwandt/ElePose) |

## Comparisons of human mesh recovery methods on Human3.6M and 3DPW.
| Method         | Publication | Highlight                          | Human3.6M MPJPE↓ | Human3.6M PA-MPJPE↓ | 3DPW MPJPE↓ | 3DPW PA-MPJPE↓ | 3DPW PVE↓ | Code                                                                                |
|----------------|-------------|------------------------------------|------------------|---------------------|-------------|----------------|-----------|-------------------------------------------------------------------------------------|
| [VirtualMarker](http://openaccess.thecvf.com/content/CVPR2023/html/Ma_3D_Human_Mesh_Estimation_From_Virtual_Markers_CVPR_2023_paper.html)  | CVPR'23     | novel intermediate representation | 47.3             | 32.0                | 67.5        | 41.3           | 77.9      | [Code](https://github.com/ShirleyMaxx/VirtualMarker)                                |
| [NIKI](http://openaccess.thecvf.com/content/CVPR2023/html/Li_NIKI_Neural_Inverse_Kinematics_With_Invertible_Neural_Networks_for_3D_CVPR_2023_paper.html)           | CVPR'23     | inverse kinematics                | -                | -                   | 71.3        | 40.6           | 86.6      | [Code](https://github.com/Jeff-sjtu/NIKI)                                           |
| [TORE](http://openaccess.thecvf.com/content/ICCV2023/html/Dou_TORE_Token_Reduction_for_Efficient_Human_Mesh_Recovery_with_Transformer_ICCV_2023_paper.html)           | ICCV'23     | efficient transformer             | 59.6             | 36.4                | 72.3        | 44.4           | 88.2      | [Code](https://frank-zy-dou.github.io/projects/Tore/index.html)                     |
| [JOTR](http://openaccess.thecvf.com/content/ICCV2023/html/Li_JOTR_3D_Joint_Contrastive_Learning_with_Transformers_for_Occluded_Human_ICCV_2023_paper.html)           | ICCV'23     | contrastive learning              | -                | -                   | 76.4        | 48.7           | 92.6      | [Code](https://github.com/xljh0520/JOTR)                                            |
| [HMDiff](http://openaccess.thecvf.com/content/ICCV2023/html/Foo_Distribution-Aligned_Diffusion_for_Human_Mesh_Recovery_ICCV_2023_paper.html)         | ICCV'23     | reverse diffusion processing      | 49.3             | 32.4                | 72.7        | 44.5           | 82.4      | [Code](https://gongjia0208.github.io/HMDiff/)                                       |
| [ReFit](http://openaccess.thecvf.com/content/ICCV2023/html/Wang_ReFit_Recurrent_Fitting_Network_for_3D_Human_Recovery_ICCV_2023_paper.html)          | ICCV'23     | recurrent fitting network         | 48.4             | 32.2                | 65.8        | 41.0           | -         | [Code](https://github.com/yufu-wang/ReFit)                                          |
| [PyMAF-X](https://ieeexplore.ieee.org/abstract/document/10113183/)        | TPAMI'23    | regression-based one-stage whole body      | -                | -                   | 74.2        | 45.3           | 87.0      | [Code](https://www.liuyebin.com/pymaf-x/)                                  |
| [PointHMR](http://openaccess.thecvf.com/content/CVPR2023/html/Kim_Sampling_Is_Matter_Point-Guided_3D_Human_Mesh_Reconstruction_CVPR_2023_paper.html)       | CVPR'23     | vertex-relevant feature extraction         | 48.3             | 32.9                | 73.9        | 44.9           | 85.5      | - |
| [PLIKS](https://openaccess.thecvf.com/content/CVPR2023/html/Shetty_PLIKS_A_Pseudo-Linear_Inverse_Kinematic_Solver_for_3D_Human_Body_CVPR_2023_paper.html)          | CVPR'23     | inverse kinematics                         | 47.0             | 34.5                | 60.5        | 38.5           | 73.3      | [Code](https://github.com/karShetty/PLIKS) |
| [ProPose](http://openaccess.thecvf.com/content/CVPR2023/html/Fang_Learning_Analytical_Posterior_Probability_for_Human_Mesh_Recovery_CVPR_2023_paper.html)        | CVPR'23     | learning analytical posterior probability  | 45.7             | 29.1                | 68.3        | 40.6           | 79.4      | [Code](https://github.com/NetEase-GameAI/ProPose)  |
| [POTTER](http://openaccess.thecvf.com/content/CVPR2023/html/Zheng_POTTER_Pooling_Attention_Transformer_for_Efficient_Human_Mesh_Recovery_CVPR_2023_paper.html)         | CVPR'23     | pooling attention transformer              | 56.5             | 35.1                | 75.0        | 44.8           | 87.4      | [Code](https://github.com/zczcwh/POTTER) |
| [PoseExaminer](http://openaccess.thecvf.com/content/CVPR2023/html/Liu_PoseExaminer_Automated_Testing_of_Out-of-Distribution_Robustness_in_Human_Pose_and_CVPR_2023_paper.html)   | ICCV'23     | automated testing of out-of-distribution   | -                | -                   | 74.5        | 46.5           | 88.6      | [Code](https://github.com/qihao067/PoseExaminer) |
| [MotionBERT](http://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html)     | ICCV'23     | pretrained human representations           | 43.1             | 27.8                | 68.8        | 40.6           | 79.4      | [Code](https://motionbert.github.io/) |
| [3DNBF](http://openaccess.thecvf.com/content/ICCV2023/html/Zhang_3D-Aware_Neural_Body_Fitting_for_Occlusion_Robust_3D_Human_Pose_ICCV_2023_paper.html)          | ICCV'23     | analysis-by-synthesis approach             | -                | -                   | 88.8        | 53.3           | -         | [Code](https://github.com/edz-o/3DNBF) |
| [FastMETRO](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_20)      | ECCV'22     | efficient architecture                     | 52.2             | 33.7                | 73.5        | 44.6           | 84.1      | [Code](https://github.com/postech-ami/FastMETRO) |
| [CLIFF](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_34)          | ECCV'22     | multi-modality inputs                      | 47.1             | 32.7                | 69.0        | 43.0           | 81.2      | [Code](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) |
| [PARE](http://openaccess.thecvf.com/content/ICCV2021/html/Kocabas_PARE_Part_Attention_Regressor_for_3D_Human_Body_Estimation_ICCV_2021_paper.html)          | ICCV'21     | part-driven attention                      | -                | -                   | 74.5        | 46.5           | 88.6      | [Code](https://pare.is.tue.mpg.de/) |
| [Graphormer](http://openaccess.thecvf.com/content/ICCV2021/html/Lin_Mesh_Graphormer_ICCV_2021_paper.html)     | ICCV'21     | GCNN-reinforced transformer                | 51.2             | 34.5                | 74.7        | 45.6           | 87.7      | [Code](https://github.com/microsoft/MeshGraphormer) |
| [PSVT](http://openaccess.thecvf.com/content/CVPR2023/html/Qiu_PSVT_End-to-End_Multi-Person_3D_Pose_and_Shape_Estimation_With_Progressive_CVPR_2023_paper.html)           | CVPR'23     | spatio-temporal encoder                    | -                | -                   | 73.1        | 43.5           | 84.0      | - |
| [GLoT](http://openaccess.thecvf.com/content/CVPR2023/html/Shen_Global-to-Local_Modeling_for_Video-Based_3D_Human_Pose_and_Shape_Estimation_CVPR_2023_paper.html)           | CVPR'23     | short-term and long-term temporal correlations  | 67.0        | 46.3                | 80.7        | 50.6           | 96.3      | [Code](https://github.com/sxl142/GLoT) |
| [MPS-Net](http://openaccess.thecvf.com/content/CVPR2022/html/Wei_Capturing_Humans_in_Motion_Temporal-Attentive_3D_Human_Pose_and_Shape_CVPR_2022_paper.html)        | CVPR'23     | temporally adjacent representations        | 69.4             | 47.4                | 91.6        | 54.0           | 109.6     | [Code](https://mps-net.github.io/MPS-Net/)                                                |
| [MAED](http://openaccess.thecvf.com/content/ICCV2021/html/Wan_Encoder-Decoder_With_Multi-Level_Attention_for_3D_Human_Shape_and_Pose_ICCV_2021_paper.html)           | ICCV'21     | multi-level attention                      | 56.4             | 38.7                | 79.1        | 45.7           | 92.6      | [Code](https://github.com/ziniuwan/maed)                                                  |
| [Lee et al.](http://openaccess.thecvf.com/content/ICCV2021/html/Lee_Uncertainty-Aware_Human_Mesh_Recovery_From_Video_by_Learning_Part-Based_3D_ICCV_2021_paper.html)     | ICCV'21     | uncertainty-aware                          | 58.4             | 38.4                | 92.8        | 52.2           | 106.1     | -                                                                                         |
| [TCMR](http://openaccess.thecvf.com/content/CVPR2021/html/Choi_Beyond_Static_Features_for_Temporally_Consistent_3D_Human_Pose_and_CVPR_2021_paper.html)           | CVPR'21     | temporal consistency                       | 62.3             | 41.1                | 95.0        | 55.8           | 111.3     | -                                                                                         |
| [VIBE](http://openaccess.thecvf.com/content_CVPR_2020/html/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.html)           | CVPR'20     | self-attention temporal network            | 65.6             | 41.4                | 82.9        | 51.9           | 99.1      | [Code](https://github.com/mkocabas/VIBE)                                                  |
| [ImpHMR](http://openaccess.thecvf.com/content/CVPR2023/html/Cho_Implicit_3D_Human_Mesh_Recovery_Using_Consistency_With_Pose_and_CVPR_2023_paper.html)         | CVPR'23     | implicitly imagine person in 3D space      | -                | -                   | 74.3        | 45.4           | 87.1      | -                                                                                         |
| [SGRE](http://openaccess.thecvf.com/content/ICCV2023/html/Wang_3D_Human_Mesh_Recovery_with_Sequentially_Global_Rotation_Estimation_ICCV_2023_paper.html)           | ICCV'23     | sequentially global rotation estimation    | -                | -                   | 78.4        | 49.6           | 93.3      | [Code](https://github.com/kennethwdk/SGRE)                                                |
| [PMCE](http://openaccess.thecvf.com/content/ICCV2023/html/You_Co-Evolution_of_Pose_and_Mesh_for_3D_Human_Body_Estimation_ICCV_2023_paper.html)           | ICCV'23     | pose and mesh co-evolution network         | 53.5             | 37.7                | 69.5        | 46.7           | 84.8      | [Code](https://github.com/kasvii/PMCE)                                                    |



