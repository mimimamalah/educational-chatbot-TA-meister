# Content
### Training data

This directory contains some of the training data that we used for training.

**sft_training**: Contains the training data used by SFT

**dpo_training**: Contains the training data used by DPO

### Data processing scripts

This directory also contains multiple data processing script:

**data_cleanup_m1.py**: Transforms the raw M1 data sample into a DPO dataset

**data_cleanup_stack.py**: Transforms the raw stack exchange datasets (Post.xml) into a DPO dataset by choosing preference pairs based on upvotes/downvotes

**data_merging_stack.py**: Used to merge multiple stack exchange datasets into one. The original stack exchange datasets can be found under the stack_exchange directory.

**dpo_to_sft.py**: Used to transform a DPO dataset into an SFT dataset (The "completion" field of the SFT dataset is the "chosen" sample in the DPO dataset)

**max_seq_length**: Used to find the longest sample in a dataset
