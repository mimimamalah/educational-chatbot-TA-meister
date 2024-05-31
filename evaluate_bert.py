from transformers import TrainerCallback, TrainerState, TrainerControl
import evaluate
from bert_score import score

class BERTScoreEvaluator(TrainerCallback):
  
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if tokenizer and model:
            # Assuming eval_dataset is passed and contains 'prompt', 'chosen', and 'rejected'
            eval_dataloader = kwargs['eval_dataloader']
            prompts, chosens, rejecteds = [], [], []
            
            for batch in eval_dataloader:
                prompts.extend(batch['prompt'])
                chosens.extend(batch['chosen'])
                rejecteds.extend(batch['rejected'])
            
            # Calculate BERTScore for chosen vs. rejected
            P_chosen, R_chosen, F1_chosen = score(chosens, prompts, lang="en", model_type="bert-base-uncased")
            P_rejected, R_rejected, F1_rejected = score(rejecteds, prompts, lang="en", model_type="bert-base-uncased")
            
            # Log the BERTScore metrics
            print(f"BERTScore for Chosen - Precision: {P_chosen.mean().item():.4f}, Recall: {R_chosen.mean().item():.4f}, F1: {F1_chosen.mean().item():.4f}")
            print(f"BERTScore for Rejected - Precision: {P_rejected.mean().item():.4f}, Recall: {R_rejected.mean().item():.4f}, F1: {F1_rejected.mean().item():.4f}")
            
            # Use F1 scores for rewards
            chosen_rewards = F1_chosen.mean().item()
            rejected_rewards = F1_rejected.mean().item()

            # Calculate reward margin
            reward_margin = chosen_rewards - rejected_rewards
            
            # Integrate the rewards into your training process
            control.chosen_rewards = chosen_rewards
            control.rejected_rewards = rejected_rewards
            control.reward_margin = reward_margin
