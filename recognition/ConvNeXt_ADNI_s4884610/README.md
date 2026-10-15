
## Feasibility Report:

## User Need, Scope & Acceptance Criteria: 

**Intended User:** The primary user is a clinical reviewer who is screening T1-weighted structural brain MRI data.  
**Purpose:** Acts as an assistant that can classify clear cases of AD and report low-confidence results to a specialist for expert examination.  
**Testable acceptance criteria:** Achieve accuracy of at least 0.8 on a held-out test split; must beat baseline in macro-F1 and AUROC score as well.   
**Leakage:** Must check to ensure no patient ID appears in more than one split.  
**Resources:** Train locally with testing; final testing is performed on Rangpur with less than 10GB VRAM and less than 4 hours train time per run.    
**Confidence:** Have a threshold where, if above 90% confidence, accept the result automatically, whilst ensuring that no more than 30% of the scans themselves are referred to an expert.

## Model Choice & Course Concepts:

**Dataset:** ADNI at /home/groups/comp3710/ADNI. Has two classes: AD and CN. Splits are made by the patient ID. Ensuring no leakage by the checks described above.  
**Baseline:** simple small CNN with 3-4 convolution blocks with the same test splits and preprocessing.   
**Model:** ConvNeXt: [https\://www\.geeksforgeeks.org/computer-vision/convnext/](https://www.geeksforgeeks.org/computer-vision/convnext/)  
**Related Course concepts:** CNNs, activation functions, normalisation, regularisation.   
**Confident analysis:** softmax confidence, reliability diagrams, with temperature scale fitting on the validation set as a fix for uncertainty. 

## Preliminary Feasibility Evidence: 

20 steps of the implemented torch cnn took 0.132 secs per stage, with 4.7 gb total vram. At batch size of 32, it is approximately 40 seconds of compute time for 10,000 training images. With both the training for AD and NC 20,000 training images should take approximately 80 seconds per epoch, or 50 epochs per hour. Which is well within the bounds set above.  
The following test/train split was found in the data: AD\_NC/test/NC **4540**, AD\_NC/test/AD **4460**, AD\_NC/train/NC **11120**, AD\_NC/train/AD **10400\.**  
Image size from testing is 256x240(W, H) and is a grayscale image.  
No overlap between patient id’s is confirmed from initial testing.

## Risks, Budget & Fallback:

**Rangpur queue delays:** run models locally as I have access to a 16GB VRAM GPU locally.  
**Patient leakage:** testing on Rangpur confirms that there is no overlap between train and test splits.  
**Overconfident wrong predictions:** temperature scaling to adjust confidence score while keeping the accuracy and classification itself the same.  
**Next experiment:** train the baseline CNN to convergence and a smaller version of the ConvNeXt and plot the reliability diagram of each.   
**Fallback:** Run the ConvNeXt with fewer blocks, narrower stages and reduced input resolution. The pipeline itself is the same so we ensure there is always something runnable if we simplify the model but keep the same structure.   
**Budget:** From before, 50 epochs per hour for the ConvNeXt with only 4.7GB VRAM; call it 10 runs total, 50-100 epochs per test; that gives around 10 to 20 GPU hours total.
