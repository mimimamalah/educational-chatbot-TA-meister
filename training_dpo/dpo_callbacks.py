# callbacks.py
from transformers import TrainerCallback
import wandb
from datasets import load_metric

class MultiMetricCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.comet_metric = load_metric("comet")
        self.bleu_metric = load_metric("bleu")
        self.bertscore_metric = load_metric("bertscore")
        self.bleurt_metric = load_metric("bleurt")
        self.meteor_metric = load_metric("meteor")

    def on_evaluate(self, args, state, control, **kwargs):
        logs = {}
        eval_dataloader = kwargs['eval_dataloader']
        outputs = [example['chosen'] for example in eval_dataloader.dataset]
        references = [example['rejected'] for example in eval_dataloader.dataset]

        # Compute metrics
        comet_results = self.comet_metric.compute(predictions=outputs, references=references)
        bleu_results = self.bleu_metric.compute(predictions=outputs, references=references)
        bertscore_results = self.bertscore_metric.compute(predictions=outputs, references=references, lang="en")
        bleurt_results = self.bleurt_metric.compute(predictions=outputs, references=references)
        meteor_results = self.meteor_metric.compute(predictions=outputs, references=references)
        
        # Combine all metrics
        logs.update(comet_results)
        logs.update(bleu_results)
        logs.update(bertscore_results)
        logs.update(bleurt_results)
        logs.update(meteor_results)
        
        # Log to WandB
        wandb.log(logs, step=state.global_step)
        return control