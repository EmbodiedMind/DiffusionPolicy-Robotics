# Diffusion Policy in Robotic Manipulation
This repository contains a collection of resources and papers on ***Diffusion Models*** for ***Robotic Manipulation***.

🚀 Please check out our survey paper [ Diffusion Policy in Robotic Manipulation](https://arxiv.org/abs/2311.01223)

![image info](./timeline.png)

## 📑Table of Contents
- 🤖[Diffusion Policy in Robotic Manipulation](#diffusion-policy-in-robotic-manipulation)
  - 📑[Table of Contents](#table-of-contents)
  - 📖[Papers](#papers)
    - 📊[Data Representation](#data-representation)
      - [2D Representation](#2d-representation)
      - [3D Representation](#3d-representation)
      - [Heterogeneous Data](#heterogeneous-data)
    - 🧠[Model Architecture](#model-architecture)
      - [Large Language Model + Diffusion](#large-language-model--diffusion)
      - [Small Size CNN or Transformer Model + Diffusion](#small-size-cnn-or-transformer-model--diffusion)
      - [VAE / VQ-VAE + Diffusion](#vae--vq-vae--diffusion)
    - 🌊[Diffusion Strategy](#diffusion-strategy)
      - [Incorporating Reinforcement Learning](#incorporating-reinforcement-learning)
      - [Diffusion Model + Equivariance](#diffusion-model--equivariance)
      - [Accelerated Sampling or Denoising Strategies](#accelerated-sampling-or-denoising-strategies)
      - [Employing Classifier (free) Guidance](#employing-classifier-free-guidance)
      - [Integration with Self-Supervised Learning](#integration-with-self-supervised-learning)
  - 📜[Citation](#citation)

##  📖Papers
### 📊Data Representation
#### 2D Representation

- **LATENT ACTION PRETRAINING FROM VIDEOS**, ICLR 2025. [[paper](https://arxiv.org/abs/2410.11758)] [[code](https://github.com/LatentActionPretraining/LAPA)]
- **Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation**, ICLR 2024. [[paper](https://arxiv.org/abs/2312.13139)] [[code](https://github.com/bytedance/GR-1)]
- **Human2Robot: Learning Robot Actions from Paired Human-Robot Videos**, arXiv 2025. [[paper](https://arxiv.org/abs/2502.16587)] [[code](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)]
- **Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation**, CVPR 2025. [[paper](https://arxiv.org/abs/2406.14235)] [[code](https://github.com/aCodeDog)]
- **Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation**, arXiv 2502. [[paper](https://arxiv.org/abs/2502.20391)] [[code](https://github.com/siddhanthaldar/Point-Policy)]
- **Learning an Actionable Discrete Diffusion Policy via Large-Scale Actionless Video Pre-Training**, NeurIPS 2024. [[paper](https://arxiv.org/abs/2402.14407)] [[code](https://github.com/tinnerhrhe/VPDD)]


#### 3D Representation

- **Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers**, NeurIPS 2024. [[paper](https://arxiv.org/abs/2409.20537)] [[code](https://github.com/liruiw/HPT)]
- **PoCo: Policy Composition from and for Heterogeneous Robot Learning**, RSS 2024. [[paper](https://arxiv.org/abs/2402.02511)]
- **Universal Actions for Enhanced Embodied Foundation Models**, arXiv 2025. [[paper](https://arxiv.org/abs/2501.10105)] [[code](https://github.com/2toinf/UniAct)]


#### Heterogeneous Data

- **GenDP: 3D Semantic Fields for Category-Level Generalizable Diffusion Policy**, CoRL 2024. [[paper](https://arxiv.org/abs/2410.17488)] [[code](https://github.com/WangYixuan12/gild)]
- **Generalizable Humanoid Manipulation with Improved 3D Diffusion Policies**, arXiv 2024. [[paper](https://arxiv.org/abs/2410.10803)] [[code](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy)]
- **3D Diffusion Policy**, RSS  2024. [[paper](https://arxiv.org/abs/2403.03954)] [[code](https://github.com/YanjieZe/3D-Diffusion-Policy)]
- **3D Diffuser Actor: Policy Diffusion with 3D Scene Representations**, CoRL 2024. [[paper](https://arxiv.org/abs/2402.10885)] [[code](https://github.com/nickgkan/3d_diffuser_actor)]
- **ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation**, CoRL 2023. [[paper](https://arxiv.org/abs/2311.01223)] [[code](https://github.com/zhouxian/act3d-chained-diffuser)]
- **ADAMANIP: ADAPTIVE ARTICULATED OBJECT MANIPULATION ENVIRONMENTS AND POLICY LEARNING**, ICLR 2025. [[paper](https://arxiv.org/abs/2502.11124)]
- **Shelving, Stacking, Hanging: Relational Pose Diffusion for Multi-modal Rearrangement**, CoRL 2023. [[paper](https://arxiv.org/abs/2307.04751)] [[code](https://github.com/anthonysimeonov/rpdiff)]
- **StructDiffusion: Object-Centric Diffusion for Semantic Rearrangement of Novel Objects**, RSS 2023. [[paper](https://arxiv.org/abs/2211.04604)] [[code](https://github.com/StructDiffusion/StructDiffusion)]
- **DexGrasp-Diffusion: Diffusion-based Unified Functional Grasp Synthesis Method for Multi-Dexterous Robotic Hands**, arXiv 2024. [[paper](https://arxiv.org/abs/2407.09899)] [[code](https://github.com/showlab/Awesome-Robotics-Diffusion)]
- **DexDiffuser: Generating Dexterous Grasps with Diffusion Models**, arXiv 2402. [[paper](https://arxiv.org/abs/2402.02989)] [[code](https://github.com/YuLiHN/DexDiffuser)]
- **ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation**, arXiv 2024. [[paper](https://arxiv.org/abs/2406.01586)] [[code](https://github.com/ManiCM-fast/ManiCM)]
- **Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation**, CVPR 2024. [[paper](https://arxiv.org/abs/2403.03890)] [[code](https://github.com/dyson-ai/hdp)]
- **DNAct: Diffusion Guided Multi-Task 3D Policy Learning**, arXiv 2403. [[paper](https://arxiv.org/abs/2403.04115)]
- **EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning**, CoRL 2024. [[paper](https://arxiv.org/abs/2407.01479)] [[code](https://github.com/yjy0625/equibot)]
- **EQUIVARIANT DESCRIPTION FIELDS: SE(3)-EQUIVARIANT ENERGY-BASED MODELS FOR END-TO-END VISUAL ROBOTIC MANIPULATION LEARNING**, ICLR 2023. [[paper](https://arxiv.org/abs/2206.08321)] [[code](https://github.com/tomato1mule/edf)]
- **RoboKeyGen: Robot Pose and Joint Angles Estimation via Diffusion-based 3D Keypoint Generation**, ICRA 2024. [[paper](https://arxiv.org/abs/2403.18259)] [[code](https://github.com/Nimolty/RoboKeyGen)]
- **Bi3D Diffuser Actor: 3D Policy Diffusion for Bi-manual Robot Manipulation**, CoRL 2024.


### 🧠Model Architecture

#### Large Language Model + Diffusion

- **Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals 和 Multimodal Diffusion Transformer for Learning from Play**, RSS 2024. [[paper](https://arxiv.org/abs/2407.05996)] [[code](https://github.com/intuitive-robots/mdt_policy)]
- **ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model**, arXiv 2025. [[paper](https://arxiv.org/abs/2502.14420)]
- **TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation**, arXiv 2025. [[paper](https://arxiv.org/abs/2409.12514)]
- **RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation**, arXiv 2025. [[paper](https://arxiv.org/abs/2410.07864)] [[code](https://github.com/thu-ml/RoboticsDiffusionTransformer)]
- **Scaling Robot Learning with Semantically Imagined Experience**, RSS 2023. [[paper](https://arxiv.org/abs/2302.11550)]
- **LATENT ACTION PRETRAINING FROM VIDEOS**, ICLR 2025. [[paper](https://arxiv.org/abs/2410.11758)] [[code](https://github.com/LatentActionPretraining/LAPA)]


#### Small Size CNN or Transformer Model + Diffusion

- **Octo**, RSS 2024. [[paper](https://arxiv.org/abs/2405.12213)] [[code](https://github.com/octo-models/octo)]
- **ALOHA Unleashed: a transformer-based learning architecture trained with a diffusion loss**, CoRL 2024. [[paper](https://arxiv.org/abs/2410.13126)] [[code](https://github.com/aloha-unleashed/aloha_unleashed)]
- **S2-Diffusion: Generalizing from Instance-level to Category-level Skills in Robot Manipulation(S^2)**, arXiv 2025. [[paper](https://arxiv.org/abs/2502.09389)] [[code](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)]
- **Planning with Diffusion for Flexible Behavior Synthesis**, ICML 2022. [[paper](https://arxiv.org/abs/2205.09991)] [[code](https://github.com/jannerm/diffuser)]
- **Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models**, IROS 2023. [[paper](https://arxiv.org/abs/2308.01557)] [[code](https://github.com/jacarvalho/mpd-public)]
- **Diffusion Policy for Collision Avoidance in a Two-Arm Robot Setup**, arXiv 2024. [[code](https://github.com/showlab/Awesome-Robotics-Diffusion)]
- **SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion**, ICRA 2023. [[paper](https://arxiv.org/abs/2209.03855)] [[code](https://github.com/robotgradient/grasp_diffusion)]
- **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion（不确定）**, arXiv 2023. [[paper](https://arxiv.org/abs/2303.04137)] [[code](https://github.com/real-stanford/diffusion_policy)]
- **3D Diffusion Policy**, RSS 2024. [[paper](https://arxiv.org/abs/2403.03954)] [[code](https://github.com/YanjieZe/3D-Diffusion-Policy)]
- **Scaling Robot Learning with Semantically Imagined Experience**, arXiv 2023. [[paper](https://arxiv.org/abs/2302.11550)] [[code](https://github.com/YanjieZe/Paper-List/blob/main/topics/robotic_manipulation.md)]
- **3D Diffuser Actor: Policy Diffusion with 3D Scene Representations**, CoRL 2024. [[paper](https://arxiv.org/abs/2402.10885)] [[code](https://github.com/nickgkan/3d_diffuser_actor)]
- **PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play**, CoRL 2023. [[paper](https://arxiv.org/abs/2312.04549)] [[code](https://github.com/shikharbahl/playfusion_dataset)]
- **Diffusion Trajectory-guided Policy for Long-horizon Robot Manipulation**, arXiv 2502. [[paper](https://arxiv.org/abs/2502.10040)] [[code](https://github.com/mbreuss/diffusion-literature-for-robotics)]
- **Diffusion Model-Augmented Behavioral Cloning**, arXiv 2023. [[paper](https://arxiv.org/abs/2302.13335)] [[code](https://github.com/NTURobotLearningLab/dbc)]
- **Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition**, CoRL 2023. [[paper](https://arxiv.org/abs/2307.14535)] [[code](https://github.com/real-stanford/scalingup)]
- **C3DM: Constrained-Context Conditional Diffusion Models for Imitation Learning**, TMLR 2024. [[paper](https://arxiv.org/abs/2311.01419)] [[code](https://github.com/showlab/Awesome-Robotics-Diffusion)]
- **Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models**, CoRL 2023. [[paper](https://arxiv.org/abs/2401.03360)] [[code](https://github.com/generative-skill-chaining/gsc-code)]
- **Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning**, arXiv 2405. [[paper](https://arxiv.org/abs/2405.18196)] [[code](https://github.com/vv19/rendiff)]
- **ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation**, CoRL 2023. [[code](https://github.com/zhouxian/act3d-chained-diffuser)]
- **RK-Diffuser（Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation）**, CVPR 2024. [[paper](https://arxiv.org/abs/2403.03890)] [[code](https://github.com/dyson-ai/hdp)]
- **The Ingredients for Robotic Diffusion Transformers（DiT-Block Policy）**, arXiv 2410. [[code](https://github.com/sudeepdasari/dit-policy)]
- **MTDP: Modulated Transformer Diffusion Policy Model**, arXiv 2502. [[paper](https://arxiv.org/abs/2502.09029)]
- **StructDiffusion: Object-Centric Diffusion for Semantic Rearrangement of Novel Objects**, RSS 2023. [[paper](https://arxiv.org/abs/2211.04604)] [[code](https://github.com/StructDiffusion/StructDiffusion)]


#### VAE / VQ-VAE + Diffusion

- **SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions in Diffusion-Based Task Execution**, arXiv 2024. [[code](https://github.com/Liang-ZX/SkillDiffuser)]
- **GEVRM：Goal-Expressive Video Generation Model For Robust Visual Manipulation**, ICLR 2025. [[paper](https://arxiv.org/abs/2502.09268)]
- **Discrete Policy: Learning Disentangled Action Space for Multi-Task Robotic Manipulation**, ICRA 2025. [[paper](https://arxiv.org/abs/2409.18707)]
- **Learning an Actionable Discrete Diffusion Policy via Large-Scale Actionless Video Pre-Training**, NeurIPS 2024. [[paper](https://arxiv.org/abs/2402.14407)] [[code](https://github.com/tinnerhrhe/VPDD)]
- **LATENT ACTION PRETRAINING FROM VIDEOS**, ICLR 2025. [[paper](https://arxiv.org/abs/2410.11758)] [[code](https://github.com/LatentActionPretraining/LAPA)]



### 🌊Diffusion Strategy

#### Incorporating Reinforcement Learning

- **IMITATING HUMAN BEHAVIOUR WITH DIFFUSION MODELS**, ICLR 2023. [[paper](https://arxiv.org/abs/2301.10677)] [[code](https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion)]
- **Goal-Conditioned Imitation Learning using Score-based Diffusion Policies**, RSS 2023. [[paper](https://arxiv.org/abs/2304.02532)] [[code](https://github.com/intuitive-robots/beso)]
- **Imit Diff: Semantics Guided Diffusion Transformer with Dual Resolution Fusion for Imitation Learning**, arXiv 2502. [[paper](https://arxiv.org/abs/2502.09649)]
- **ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation**, arXiv 2406. [[paper](https://arxiv.org/abs/2406.01586)] [[code](https://github.com/ManiCM-fast/ManiCM)]
- **Seed up the inference phase of diffusion models by hierarchical sampling（有问题*）**, arXiv 2013. [[code](https://github.com/FilippoMB/Diffusion_models_tutorial/blob/main/diffusion_from_scratch.ipynb)]
- **DiffuserLite: Towards Real-time Diffusion Planning**, NeurIPS 2024. [[paper](https://arxiv.org/abs/2401.15443)] [[code](https://github.com/diffuserlite/diffuserlite.github.io)]
- **ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation**, CoRL 2023. [[code](https://github.com/zhouxian/act3d-chained-diffuser)]
- **RK-Diffuser：Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation**, CVPR 2024. [[paper](https://arxiv.org/abs/2403.03890)]


#### Diffusion Model + Equivariance

- **Diffusion Policy Policy Optimization**, ICLR 2025. [[paper](https://arxiv.org/abs/2409.00588)] [[code](https://github.com/irom-princeton/dppo)]
- **Goal-Conditioned Imitation Learning using Score-based Diffusion Policies**, RSS 2023. [[paper](https://arxiv.org/abs/2304.02532)] [[code](https://github.com/intuitive-robots/beso)]
- **RoboKeyGen: Robot Pose and Joint Angles Estimation via Diffusion-based 3D Keypoint Generation**, ICRA 2024. [[paper](https://arxiv.org/abs/2403.18259)]
- **ReorientDiff: Diffusion Model based Reorientation for Object Manipulation**, LEAP 2023. [[paper](https://arxiv.org/abs/2303.12700)] [[code](https://github.com/mbreuss/diffusion-literature-for-robotics)]
- **Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation**, CoRL 2024. [[paper](https://arxiv.org/abs/2410.14868)]


#### Accelerated Sampling or Denoising Strategies

- **Dynamics-Guided Diffusion Model for Sensor-less Robot Manipulator Design**, CoRL 2024. [[paper](https://arxiv.org/abs/2402.15038)]
- **SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions in Diffusion-Based Task Execution**, CVPR 2024. [[paper](https://arxiv.org/abs/2312.11598)] [[code](https://github.com/Liang-ZX/SkillDiffuser)]
- **Goal-Conditioned Imitation Learning using Score-based Diffusion Policies**, RSS 2023. [[paper](https://arxiv.org/abs/2304.02532)] [[code](https://github.com/intuitive-robots/beso)]
- **ReorientDiff: Diffusion Model based Reorientation for Object Manipulation**, ICRA 2024. [[paper](https://arxiv.org/abs/2303.12700)] [[code](https://github.com/UtkarshMishra04/ReorientDiff)]
- **Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models**, CoRL 2023. [[paper](https://arxiv.org/abs/2401.03360)] [[code](https://github.com/generative-skill-chaining/gsc-code)]


#### Employing Classifier (free) Guidance

- **SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion**, arXiv 2209. [[paper](https://arxiv.org/abs/2209.03855)] [[code](https://github.com/robotgradient/grasp_diffusion)]
- **Planning with Diffusion for Flexible Behavior Synthesis**, ICML 2022. [[paper](https://arxiv.org/abs/2205.09991)] [[code](https://github.com/jannerm/diffuser)]


#### Integration with Self-Supervised Learning

- **EDGI: Equivariant Diffusion for Planning with Embodied Agents**, NeurIPS 2023. [[paper](https://arxiv.org/abs/2303.12410)]
- **Diffusion Policy Policy Optimization**, arXiv 2409. [[paper](https://arxiv.org/abs/2409.00588)] [[code](https://github.com/irom-princeton/dppo)]
- **Diffusion Reward: Learning Rewards via Conditional Video Diffusion**, ECCV 2024. [[paper](https://arxiv.org/abs/2312.14134)] [[code](https://github.com/TEA-Lab/diffusion_reward)]
- **Planning with Diffusion for Flexible Behavior Synthesis**, ICML 2022. [[paper](https://arxiv.org/abs/2205.09991)] [[code](https://github.com/jannerm/diffuser)]


### Uncategorized

#### Others

- **Sparse Diffusion Policy: A Sparse, Reusable, and Flexible Policy for Robot Learning**, CoRL 2024. [[paper](https://arxiv.org/abs/2407.01531)] [[code](https://github.com/AnthonyHuo/SDP)]
- **ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model**, arXiv 2502. [[paper](https://arxiv.org/abs/2502.14420)]
- **Generate Subgoal Images before Act: Unlocking the Chain-of-Thought Reasoning in Diffusion Model for Robot Manipulation with Multi-modal Prompts**, CVPR 2024. [[paper](https://arxiv.org/abs/2310.09676)]
- **Plan Diffuser: Grounding LLM Planners with Diffusion Models for Robotic Manipulation**, ICRA 2024.
- **Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation**, arXiv 2502. [[paper](https://arxiv.org/abs/2502.16707)]
- **Subgoal Diffuser: Coarse-to-fine Subgoal Generation to Guide Model Predictive Control for Robot Manipulation**, ICRA 2024. [[paper](https://arxiv.org/abs/2403.13085)]
- **ZERO-SHOT ROBOTIC MANIPULATION WITH PRETRAINED IMAGE-EDITING DIFFUSION MODELS**, arXiv 2310. [[paper](https://arxiv.org/abs/2310.10639)] [[code](https://github.com/kvablack/susie)]
- **Learning Universal Policies via Text-Guided Video Generation**, NeurIPS 2023. [[paper](https://arxiv.org/abs/2302.00111)]
- **Diffusion models to generate rich synthetic expert data（AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners）**, ICML 2023. [[paper](https://arxiv.org/abs/2302.01877)]
- **DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning**, arXiv 2025. [[paper](https://arxiv.org/abs/2502.16932)]
- **JUICER: Data-Efficient Imitation Learning for Robotic Assembly**, IROS 2024. [[paper](https://arxiv.org/abs/2404.03729)]
- **ALDM-Grasping: Diffusion-aided Zero-Shot Sim-to-Real Transfer for Robot Grasping**, IROS 2024. [[paper](https://arxiv.org/abs/2403.11459)] [[code](https://github.com/levyisthebest/ALDM-grasping)]
- **Scaling Robot Learning with Semantically Imagined Experience**, RSS 2023. [[paper](https://arxiv.org/abs/2302.11550)]
- **DexGrasp-Diffusion: Diffusion-based Unified Functional Grasp Synthesis Method for Multi-Dexterous Robotic Hands**, arXiv 2407. [[paper](https://arxiv.org/abs/2407.09899)]
- **DexDiffuser: Generating Dexterous Grasps with Diffusion Models**, arXiv 2024. [[paper](https://arxiv.org/abs/2402.02989)] [[code](https://github.com/YuLiHN/DexDiffuser)]
- **Diffusion-EDFs: Bi-equivariant Denoising Generative Modeling on SE(3) for Visual Robotic Manipulation**, CVPR 2024. [[paper](https://arxiv.org/abs/2309.02685)] [[code](https://github.com/tomato1mule/diffusion_edf)]
- **EDGI: Equivariant Diffusion for Planning with Embodied Agents**, NeurIPS 2023. [[paper](https://arxiv.org/abs/2303.12410)]
- **EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning**, CoRL 2024. [[paper](https://arxiv.org/abs/2407.01479)] [[code](https://github.com/yjy0625/equibot)]
- **EquiDiff: A Conditional Equivariant Diffusion Model For Trajectory Prediction**, ITSC 2023. [[paper](https://arxiv.org/abs/2308.06564)] [[code](https://github.com/apexrl/Diff4RLSurvey)]
- **Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals**, RSS 2024. [[paper](https://arxiv.org/abs/2407.05996)] [[code](https://github.com/intuitive-robots/mdt_policy)]
- **Multimodal Diffusion Transformer for Learning from Play**, arXiv 2023.
- **Crossway Diffusion: Improving Diffusion-based Visuomotor Policy via Self-supervised Learning**, ICRA 2024. [[paper](https://arxiv.org/abs/2307.01849)]
- **Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning**, arXiv 2206. [[paper](https://arxiv.org/abs/2206.08321)] [[code](https://github.com/tomato1mule/edf)]
- **SE(3)-Equivariant Relational Rearrangement with Neural Descriptor Fields**, arXiv 2211. [[paper](https://arxiv.org/abs/2211.09786)] [[code](https://github.com/anthonysimeonov/relational_ndf)]
- **Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation**, ICRA 2022. [[paper](https://arxiv.org/abs/2112.05124)] [[code](https://github.com/anthonysimeonov/ndf_robot)]
- **Edge Grasp Network: A Graph-Based SE(3)-invariant Approach to Grasp Detection**, ICRA 2023. [[paper](https://arxiv.org/abs/2211.00191)] [[code](https://github.com/HaojHuang/Edge-Grasp-Network)]
- **Local Neural Descriptor Fields: Locally Conditioned Object Representations for Manipulation**, ICRA 2023. [[paper](https://arxiv.org/abs/2302.03573)] [[code](https://github.com/elchun/lndf_robot)]
- **Robot Manipulation Task Learning by Leveraging SE(3) Group Invariance and Equivariance**, arXiv 2023. [[paper](https://arxiv.org/abs/2308.14984)]
- **Geometric Algebra Transformer**, NeurIPS 2023. [[paper](https://arxiv.org/abs/2305.18415)] [[code](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer)]
- **Implicit Behavioral Cloning**, CoRL 2021. [[paper](https://arxiv.org/abs/2109.00137)] [[code](https://github.com/google-research/ibc)]
- **EquivAct: SIM(3)-Equivariant Visuomotor Policies beyond Rigid Object Manipulation**, ICRA 2024. [[paper](https://arxiv.org/abs/2310.16050)]
- **EquiDiff: A Conditional Equivariant Diffusion Model For Trajectory Prediction**, arXiv 2023. [[paper](https://arxiv.org/abs/2308.06564)]
- **Symmetric Models for Visual Force Policy Learning**, ICRA 2024. [[paper](https://arxiv.org/abs/2308.14670)]
- **Compositional Foundation Models for Hierarchical Planning **, NeurIPS 2023. [[paper](https://arxiv.org/abs/2309.08587)] [[code](https://github.com/anuragajay/hip)]
- **DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics**, arXiv 2023. [[paper](https://arxiv.org/abs/2210.02438)]
- **Universal Actions for Enhanced Embodied Foundation Models**, CVPR 2025. [[paper](https://arxiv.org/abs/2501.10105)] [[code](https://github.com/2toinf/UniAct)]
- **GenAug: Retargeting behaviors to unseen situations via Generative Augmentation **, RSS 2023. [[paper](https://arxiv.org/abs/2302.06671)] [[code](https://github.com/genaug)]
- **CACTI: A Framework for Scalable Multi-Task Multi-Scene Visual Imitation Learning **, arXiv 2212. [[paper](https://arxiv.org/abs/2212.05711)] [[code](https://github.com/cacti-framework/cacti-framework.github.io)]
- **CACTI: A Framework for Scalable Multi-Task Multi-Scene Visual Imitation Learning **, arXiv 2410. [[paper](https://arxiv.org/abs/2410.24164)]

## Citation
```
@article{,
  title={Diffusion Policy in Robotic Manipulation},
  author={},
  journal={arXiv preprint },
  year={2025}
}
```
