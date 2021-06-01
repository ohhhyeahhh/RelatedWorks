
## 2020年
### CVPR
- **SiamCAR**: Siamese Fully Convolutional Classification and Regression for Visual Tracking [[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_SiamCAR_Siamese_Fully_Convolutional_Classification_and_Regression_for_Visual_Tracking_CVPR_2020_paper.html)][[code](https://github.com/ohhhyeahhh/SiamCAR)]
- **SiamBAN**: Siamese Box Adaptive Network for Visual Tracking. [[code](https://github.com/hqucv/siamban)]
- **D3S** – A Discriminative Single Shot Segmentation Tracker. [[paper](https://arxiv.org/abs/1911.08862)]
- **MAML**: Tracking by Instance Detection: A Meta-Learning Approach. [[paper](https://arxiv.org/abs/2004.00830)]
- **SiamAttn**: Deformable Siamese Attention Networks for Visual Object Tracking.
- **CGACD**: Correlation-Guided Attention for Corner Detection Based Visual Tracking.
- **Siam R-CNN**: Visual Tracking by Re-Detection.[[code](http://www.vision.rwth-aachen.de/page/siamrcnn)]
'''
将Faster-RCNN 结合到 Siamese系列跟踪,用RPN网络提取出ROI，将当前帧所有ROI与第一帧的GT的ROI cat进行re-detection，选定得分较高的boxes，再与上一帧的boxes两两组合（距离满足条件），再次Re-detection，得到更精确的boxes(有相似目标就有几率会有多个box)。将检测到的目标通过关联的方式形成跟踪轨迹链，如果一但有干扰目标存在，那么开辟一条新轨迹，最后得到多条轨迹，根据相邻轨迹之间首尾，尾首目标中心之间的距离来判断是否是同一条轨迹，最终得到目标的最终轨迹。
优点：long-term的测试效果则非常好
缺点：由于re-detection采用级联RCNN，精度高，但速度低
'''

- **PrDiMP**: Probabilistic Regression for Visual Tracking. [[code](https://github.com/visionml/pytracking)]
- Recursive Least-Squares Estimator-Aided Online Learning for Visual Tracking. [[code](https://github.com/Amgao/RLS-RTMDNet)]
- **ROAM**: Recurrently Optimizing Tracking Model. [[code](https://github.com/skyoung/ROAM)]
- One-Shot Adversarial Attacks on Visual Tracking With Dual Attention.
- **AutoTrack**: Towards High-Performance Visual Tracking for UAV With Automatic Spatio-Temporal Regularization. [[code](https://github.com/vision4robotics/AutoTrack)]
- High-Performance Long-Term Tracking With Meta-Updater. [[code](https://github.com/Daikenan/LTMU)]
长时跟踪
- **Cooling-Shrinking Attack**: Blinding the Tracker with Imperceptible Noises. [[code](https://github.com/MasterBin-IIAU/CSA)]
目标跟踪鲁棒性研究
- **MAST**: A Memory-Augmented Self-Supervised Tracker. [[code](https://github.com/zlai0/MAST)]

### ECCV
- Learning Feature Embeddings for Discriminant Model based Tracking. 
- **CLNet**: A Compact Latent Network for Fast Adjusting Siamese Tracker. 
- **Ocean**: Learning Object-aware Anchor-free Networks for Real-time Object Tracking. [[code](https://github.com/researchmm/TracKit)]
- **PG-Net**: Pixel to Global Matching Network for
Visual Tracking. 
- **Know Your Surroundings**: Exploiting Scene Information for Object Tracking.
- **SPARK**: Spatial-aware Online Incremental Attack Against Visual Tracking.
- Efficient Adversarial Attacks for Visual Object Tracking.

### AAAI
- Discriminative and Robust Online Learning for Siamese Visual Tracking.
- Exploiting Spatial Invariance for Scalable Unsupervised Object Tracking.
- **GlobalTrack**: A Simple and Strong Baseline for Long-term Tracking.
- **SiamFC++**: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines
- Real-Time Object Tracking via Meta-Learning: Efficient Model Adaptation and One-Shot Channel Pruning
- **SPSTracker**: Sub-Peak Suppression of Response Map for Robust Object Tracking

## 2019年

### CVPR
- Unsupervised Deep Tracking. [[code](https://github.com/594422814/UDT)]
- Target-Aware Deep Tracking. [[code](https://github.com/XinLi-zn/TADT)]
- **SPM-Tracker**: Series-Parallel Matching for Real-Time Visual Object Tracking. 
- **SiamRPN++**: Evolution of Siamese Visual Tracking With Very Deep Networks. [[code](https://github.com/STVIR/pysot)]
- **SiamDW**: Deeper and Wider Siamese Networks for Real-Time Visual Tracking. [[code](https://github.com/researchmm/SiamDW)]
- **GCT**:Graph Convolutional Tracking. 
- **ATOM**: Accurate Tracking by Overlap MaXimization. [[code](https://github.com/visionml/pytracking)]
- **C-RPN**: Siamese Cascaded Region Proposal Networks for Real-Time Visual Tracking. [[paper](https://arxiv.org/pdf/1812.06148.pdf)][[code](https://bitbucket.org/hengfan/crpn/src/master/)]


## ICCV
- **DiMP**: Learning Discriminative Model Prediction for Tracking. [[paper](https://arxiv.org/pdf/1904.07220.pdf)] [[code](https://github.com/visionml/pytracking)]
- **GradNet**: Gradient-Guided Network for Visual Object Tracking. [[code](https://github.com/LPXTT/GradNet-Tensorflow)]
- **UpdateNet**: Learning the Model Update for Siamese Trackers. [[code](https://github.com/zhanglichao/
updatenet)]
- **MLT**: Deep Meta Learning for Real-Time Target-Aware Visual Tracking.
- Joint Group Feature Selection and Discriminative Filter Learning for Robust Visual Object Tracking. [[code](https://github.com/XU-TIANYANG/
GFS-DCF)]
- **SPLT**: Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-Term Tracking. [[code](https://github.com/iiau-tracker/SPLT)]
- Learning Aberrance Repressed Correlation Filters for Real-Time UAV Tracking.
- Bridging the Gap Between Detection and Tracking: A Unified Approach.
