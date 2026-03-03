## 1. Abstract
Life-limiting congenital anomalies (LLCAs) require accurate prenatal diagnosis for appropriate clinical decision-making. Prenatal ultrasound (US) examinations involve multiple anatomical planes, and diagnosis depends on identifying anatomical planes and selecting diagnostically relevant planes for each anomaly. Existing automated methods either rely on plane-level annotations or aggregate heterogeneous images without modeling these diagnostic capabilities. We propose \textbf{AnomExpert}, a prototype-driven framework for prenatal US anomaly diagnosis using only case-level supervision. AnomExpert introduces learnable plane prototypes to organize unordered images into latent representations corresponding to anatomical planes without requiring plane annotations. A disease-aware sparse selection mechanism further selects diagnostically relevant planes for each anomaly. Experiments on a multi-center dataset of 3,654 cases show that AnomExpert consistently outperforms nine representative multi-instance learning methods. Using a ViT-small backbone, it achieves 86.9% accuracy and 84.2% F1-score while maintaining parameter efficiency. These findings indicate that modeling anatomical plane identification and disease-specific plane selection improves weakly supervised multi-plane prenatal US anomaly classification.

## 2. Methodology


## 3. Main Result


## 4. Train
