from transformers import TrainerCallback, TrainerState, TrainerControl
import evaluate

class BERTScoreEvaluator(TrainerCallback):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.bertscore = evaluate.load("bertscore")

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        eval_dataloader = kwargs.get('dataloader')
        metrics = {}

        for batch in eval_dataloader:
            inputs = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            outputs = model.generate(batch['input_ids'])
            predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            score = self.bertscore.compute(predictions=predictions, references=inputs, lang="en")
            metrics.update(score)

        control.metrics.update(metrics)
