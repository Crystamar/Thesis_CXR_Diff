# **CXR-Diff: a novel deep-learning framework for finding changes of interest in longitudinal chest X-rays**

<p align=center><b>Supervised by Prof. Leo Joskowicz</b></p>  
<p align=center><b>In collaboration with Dr. Michael Beil MD, Prof. Sigal Sviri Sarussi MD, Dr. Benjamin Koplewitz MD  
from Hadassah University Medical Center</b></p>  

<p align=center><b>Two paper manuscripts (1 technical & 1 clinical) in preparation</b></p>  

**Abstract**  

The assessment of changes over time in chest X-rays (CXRs) is a fundamental task in
clinical care for monitoring disease progression and treatment response. However, this
task is highly challenging due to significant variance in patient positioning, the frequent
occurrence of simultaneous, overlapping pathological conditions, the presence of medical
devices in CXRs, occluding patient anatomy, and high inter-observer variability. It is thus
of interest to develop computerized methods to automatically find the changes of interest
between pairs of CXRs (hereby current and prior CXRs).
This thesis presents a novel deep-learning framework, CXR-Diff, designed to
automatically detect and segment changes of clinical interest in CXR pairs. Our framework
does not require aligning the CXRs in a pair, does not require explicitly specifying what
constitutes a change of interest, and does not require a large set of labeled pairs of CXRs
for training, minimizing the data collection, curation, and labeling efforts by clinicians.
CXR-Diff consists of: 1) CXR-Diff-Net, a new convolutional neural network that inputs pairs
of unaligned CXRs and computes a difference map between them; 2) a model-based
pipeline that generates, from a source chest CT volume, synthetic pairs of CXRs and their
ground truth difference maps by simulating both background changes and changes of
interest. The pipeline is used to generate a configurable large, labeled dataset used to
train CXR-Diff-Net. While CXR-Diff is domain-agnostic, we validate its efficacy in the
complex environment of the Intensive Care Unit (ICU) by designing the pipeline to
specifically simulate various background and pathology changes common in the ICU.
The framework was evaluated through a series of studies. Evaluation on ICU CXR pairs
(30 pairs spanning 30 patients) benchmarked the model (trained on 60,000 synthetic
CXR pairs generated from 2,469 chest CTs) against five human clinicians. Results showed
that the model’s agreement with clinicians, measured via the Pairwise Agreement Index
(PAI), falls within the range established by the clinicians. The model achieved a sensitivity
of 0.85 to 1.00 for Positive changes and of 0.70 to 0.83 for Negative changes at high clinical
consensus levels (k ≥ 3). Technical validation on synthetic datasets demonstrated the
method’s robustness to patient positioning, and its high sensitivity across a spectrum of
magnitudes of changes of interest.
Our automated approach may improve clinical workflows in the ICU by providing a
reliable, interpretable detection of clinically relevant pathological changes.  
<br>

**Figures**  

<img width="1891" height="1037" alt="non_icu drawio" src="https://github.com/user-attachments/assets/9944a4bd-a94d-4ffa-84a8-7387b01216b3" />
(a) A healthy CXR (PA projection) of a female patient. (b) A CXR (PA
projection) with a subtle consolidation in the patient’s lower left lung  
<br><br><br>
<img width="1993" height="1047" alt="icu drawio" src="https://github.com/user-attachments/assets/498b7de7-71f8-4cab-a9c2-beb7e8919fda" />
(a) An ICU CXR (AP projection) exhibiting various lung opacities, a
large cardiac silhouette, and several medical devices, including a pacemaker, electrodes,
stickers, and sternal wires. (b) An ICU CXR (AP projection) with poor patient positioning,
resulting in artifacts obscuring the lung fields, and poorly expanded lungs.
<br><br><br>
<img width="1900" height="900" alt="icu_pair drawio" src="https://github.com/user-attachments/assets/009aa2ab-1c6f-498e-9e0a-ce1794851739" />
A longitudinal pair of ICU CXRs (AP projection),
comprising a prior (left) and a current (right), exhibiting geometric differences and
changes in medical devices.
<br><br><br>
<img width="1200" height="934" alt="DRR" src="https://github.com/user-attachments/assets/7bc821d7-5698-4610-bfbc-ddfb365b52ce" />
A synthetic CXR is generated
via numerical integration along divergent rays cast from a virtual X-ray source through a
chest CT volume.
<br><br><br>
<img width="4096" height="3235" alt="FrameworkOverviewPaper drawio" src="https://github.com/user-attachments/assets/d1ea1f6c-077e-45d9-b079-de1a6010c4c1" />
Framework overview. Overview of the CXR-Diff framework. The input is a
collection of chest CT volumes. The output is a custom CNN, CXR-Diff-Net, trained on a
large set of labeled synthetic CXR pairs. The framework consists of three steps: 1) CT
selection and volumetric preprocessing; 2) labeled synthetic CXR pairs generation; 3)
deep learning model training.
<br><br><br>
<img width="4687" height="1971" alt="PairGeneration drawio" src="https://github.com/user-attachments/assets/5248709c-8d80-4cd3-a984-a93e0323124c" />
Labeled synthetic CXR pairs generation pipeline. Overview of the
generation pipeline of labeled synthetic CXR pairs. The inputs are a source chest CT scan
and thoracic anatomical segmentations. The output is a labeled synthetic CXR pair. The
pipeline consists of six steps: 1) CT duplication; 2) synthetic entity insertion; 3) 3D
rotations; 4) DRR projection; 5) difference map computation; 6) intensity augmentations.
<br><br><br>
<img width="2758" height="1985" alt="EntityExamples1 drawio" src="https://github.com/user-attachments/assets/5b4b0275-2be3-4449-b021-3df0da500702" />
Synthetic entity examples (1). Synthetic CXR pairs from various source
chest CTs presenting: (a) An appearing consolidation in the (patient’s) right lung. (b) A
disappearing consolidation in the right lung, and a persistence in the left. (c) A
progressing consolidation in the right lung. (d) A regressing consolidation in the right
lung. (e) A persisting consolidation in the right lung. (f) An appearing pleural effusion in
the left lung in the supine position. (g) A regressing pleural effusion in both lungs in the
semi-erect position. (h) An appearing pleural effusion in both lungs in the supine position.
<br><br><br>
<img width="2767" height="1582" alt="EntityExamples2 drawio" src="https://github.com/user-attachments/assets/65e9e407-3811-4b76-8fcf-8be7653a98d9" />
Synthetic entity examples (2). Synthetic CXR pairs from various source
chest CTs presenting: (i) A progressing pneumothorax in the right lung in the supine
position. (j) An appearing subtle pneumothorax in the right lung, and a disappearing
subtle one in the left lung in the erect position. (k) An appearing fluid overload in both
lungs in the interstitial stage in the supine position. (l) A progressing fluid overload in
both lungs in the alveolar stage in the supine position. Synthetic CXRs presenting
synthetic medical devices: (m) Wires. (n) Wires and stickers. (o) Port. (p) Port. (q)
Pacemaker.
<br><br><br>
<img width="2918" height="1625" alt="Architecture drawio" src="https://github.com/user-attachments/assets/9e2500c3-ed74-48ed-ba31-f474ffb42e51" />
Model architecture. (a) An overview of the architecture of CXR-Diff-Net. The
synthetic prior and current inputs are independently encoded via two instances of a
standalone encoder. A difference representation is extracted by the Difference Encoder
module by employing two cascaded subtractions on the processed representations and
merging via addition. The difference representation is passed through a decoder, and
mapped to the range [-1, 1], resulting in an output difference map. (b) The structure of
the blocks designed for and used in the architecture.
<br><br><br>
<img width="1111" height="1171" alt="Experiments Overview drawio" src="https://github.com/user-attachments/assets/b0b9ed98-06e9-45e8-86b5-76bed4f4e74e" />
Overview of: (a) the datasets created for the experimental studies and their
grouping into categories; (b) the named deep learning models and corresponding
datasets used to train them; (c) the experimental studies including the models and
datasets employed for their execution.
<br><br><br>
<img width="3628" height="1818" alt="D_ICU drawio" src="https://github.com/user-attachments/assets/dd78d303-5e71-460e-b4e3-03c36cbe6bdc" />
Examples of synthetic CXR pairs along with their ground truth difference
maps from dataset 𝑫<sub>𝒕𝒓𝒂𝒊𝒏</sub>
<sup>𝑰𝑪𝑼</sup> . Multiple entities may appear in both the prior and current,
exhibiting different types of evolution including no change. Possible background changes
include spatial rotations, medical devices, and intensity value modifications.
<br><br><br>
<img width="1055" height="569" alt="image" src="https://github.com/user-attachments/assets/acda57ed-a3d0-4d5d-8e4e-ece7d8e54105" />
CXR Changes Labeling Software. Screenshot from the specialized CXR
change annotation software, illustrating standard annotation workflow. An ICU CXR pair
from dataset ICU_PAIRS is loaded, with both CXRs viewable in the dual pane display. Three
change annotations were arbitrarily labeled and placed on top of the current CXR. Various
functional buttons, as well as a label picker and the loaded pair’s name are located at the
upper part of the interface. To the left and right are sliders, allowing independent
modifications to the prior and current’s intensity values.
<br><br><br>
<img width="5253" height="7433" alt="PCI_fig drawio" src="https://github.com/user-attachments/assets/6cfc78fd-8cc2-402d-bddc-b2cafe1f1ce9" />
Results of Studies 4-5. Pairwise agreement index (PAI). Calculated perdetection as well as per ICU CXR pair for all pairs of annotators (A, B, C, D, E and M<sub>ICU</sub>).
The dashed lines delineate the boundary between human-human (H -H) pairs and modelhuman (M - H) pairs. Calculated over all pairs in dataset ICU_PAIRS, for Positive and
Negative changes separately, and for all changes together.
<br><br><br>
<img width="1600" height="1488" alt="SensitivityHumans drawio" src="https://github.com/user-attachments/assets/1f70a9da-3e0c-4d1b-987e-897dd1ac17d9" />
 Results of Study 4: Sensitivity of the five clinicians (A, B, C, D, and E) as a
function of their corresponding consensus levels (for 𝑘 ∈ {1,2,3,4}). Calculated over all
pairs in dataset ICU_PAIRS, for Positive and Negative changes separately.
<br><br><br>
<img width="800" height="500" alt="sensitivity_consensus_levels" src="https://github.com/user-attachments/assets/8f9f640c-79e3-4fac-8ab1-03f451cb067e" />
Results of Study 5. Sensitivity of M<sub>ICU</sub> as a function of clinician consensus
(𝑘 ∈ {1,2,3,4,5}). Calculated over all pairs in dataset ICU_PAIRS, for Positive and Negative
changes separately.
<br><br><br>
<img width="2470" height="2551" alt="AnnotationsExample drawio" src="https://github.com/user-attachments/assets/6ea68191-9c7c-4f3e-9d0a-c9c424d623b6" />
Examples of annotations. Three examples of ICU CXR pairs along with the
changes detected and labeled by each human annotators as well as model M<sub>ICU</sub>. (a) A
Positive (appearing/progressing) change is detected by all five clinicians (i.e. ∈ 𝐶≥5
𝑚𝑜𝑑𝑒𝑙)
in the patient’s lower right lung (left side of CXR). Detected by M<sub>ICU</sub> as well. A Negative
change in 𝐶≥3
𝑚𝑜𝑑𝑒𝑙 (M<sub>ICU</sub>’s consensus level 3) in the lower left lung (right side of CXR) was
undetected by the model. (b) A Positive change in 𝐶≥4
𝑚𝑜𝑑𝑒𝑙 in the patient’s lower right lung
and a Negative change in 𝐶≥5
𝑚𝑜𝑑𝑒𝑙 in the left lung, both detected by M<sub>ICU</sub>. (c) The only
Negative change in 𝐶≥5
𝑚𝑜𝑑𝑒𝑙 (left lung) undetected by M<sub>ICU</sub>

