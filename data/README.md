This directory contains some of the training data that we used for training:
dpo_train_m1.jsonl -> Contains the M1 deliverable dataset in DPO format
sft_train_m1.jsonl -> Contains the M1 deliverable dataset in SFT format 
dpo_stackexchange_43458.jsonl -> Dataset containing 43,458 DPO data samples from multiple stack exchange communities
sft_stackexchange_43043.json -> Dataset containing 43,043 SFT data samples from multiple stack exchange communities


It also contains multiple data cleanup script:
data_cleanup_m1.py -> Transforms the raw M1 data sample into a DPO dataset
data_cleanup_stack.py -> Transforms the raw stack exchange datasets (Post.xml) into a DPO dataset by choosing preference pairs based on up/downvotes
data_merging_stack.py -> Used to merge multiple stack exchange datasets into one. The original stack exchange datasets can be found under the stack_exchange directory.
dpo_to_sft.py -> Used to transform a DPO dataset into an SFT dataset (The "completion" field of the SFT dataset is the "chosen" sample in the DPO dataset)
