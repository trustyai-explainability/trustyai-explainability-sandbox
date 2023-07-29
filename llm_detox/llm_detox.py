# -*- coding: utf-8 -*-
import math
import itertools
import numpy as np
from datasets import load_dataset, DatasetDict
from scipy.spatial.distance import jensenshannon, chebyshev
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, pipeline

facebook_bart_base = "facebook/bart-base"
block_size = 128
tokenizer = BartTokenizer.from_pretrained(facebook_bart_base)
tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_function(examples):
    return tokenizer(examples["comment_text"], max_length=1024, truncation=True)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def mask_tokens(sentence, mask_token):
    masked_sentences = []
    tokens = sentence.split(' ')  # TODO: use the tokenizer
    for idx in range(len(tokens)):
        masked_sentence = tokens.copy()
        masked_sentence[idx] = mask_token
        masked_sentence = ' '.join(masked_sentence)  # TODO: use a detokenizer
        masked_sentences.append(masked_sentence)
    return masked_sentences


class MaRCo:
    base = None
    experts = []
    tokenizer = None

    def __init__(self):
        self.base = BartForConditionalGeneration.from_pretrained(facebook_bart_base, forced_bos_token_id=0)

    def load_models(self, expert_paths: list):
        for expert_path in expert_paths:
            self.experts.append(BartForConditionalGeneration.from_pretrained(expert_path))

    def train_models(self, dataset_name: str = 'jigsaw_toxicity_pred', perc: int = 20,
                     data_dir: str = 'jigsaw-toxic-comment-classification-challenge'):
        datasets = load_dataset(dataset_name, data_dir=data_dir)
        toxic_datasets = datasets.filter(lambda x: int(x['toxic']) == 1)

        td_columns = ["comment_text", 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        tokenized_datasets = toxic_datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=td_columns)

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=100,
            num_proc=4,
        )

        gminus = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

        training_args = TrainingArguments(
            "gminus-bart-large",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=gminus,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["test"],
        )

        trainer.train()

        eval_results = trainer.evaluate()
        print(f"G- perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        # train gplus model
        datasets_split = load_dataset('jigsaw_toxicity_pred', data_dir='jigsaw-toxic-comment-classification-challenge',
                                      split=['train[:' + str(perc) + '%]', 'test'])
        datasets_split = DatasetDict({
            'train': datasets_split[0],
            'test': datasets_split[1]
        }
        )

        nontoxic_datasets = datasets_split.filter(lambda x: int(x['toxic']) == 0)

        nontoxic_tokenized_datasets = nontoxic_datasets.map(tokenize_function, batched=True, num_proc=4,
                                                            remove_columns=td_columns)

        nontoxic_lm_datasets = nontoxic_tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=100,
            num_proc=4,
        )

        gplus = BartForConditionalGeneration.from_pretrained(facebook_bart_base, forced_bos_token_id=0)

        nt_training_args = TrainingArguments(
            "gplus-bart-large",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=2,
        )

        nt_trainer = Trainer(
            model=gplus,
            args=nt_training_args,
            train_dataset=nontoxic_lm_datasets["train"],
            eval_dataset=nontoxic_lm_datasets["test"],
        )

        nt_trainer.train()

        nt_eval_results = nt_trainer.evaluate()
        print(f"G+ perplexity: {math.exp(nt_eval_results['eval_loss']):.2f}")
        print('training finished')
        self.experts.append(gminus)
        self.experts.append(gplus)
        trainer.save_model('gminus')
        nt_trainer.save_model('gplus')

    def mask_toxic(self, sentence: str, threshold: float = 1, normalize: bool = True, verbose: bool = True):
        masked_sentences = mask_tokens(sentence, tokenizer.mask_token)
        distributions = []
        for model in self.experts:
            mask_substitution_scores = []
            fmp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
            for masked_sentence in masked_sentences:
                distr = fmp(masked_sentence)
                mask_substitution_score = [x['score'] for x in distr]
                mask_substitution_scores.append(mask_substitution_score)
            distributions.append(mask_substitution_scores)
        distr_pairs = itertools.combinations(distributions, 2)
        js_distances = []
        for distr_pair in distr_pairs:
            js_distance = jensenshannon(distr_pair[0], distr_pair[1], axis=1)
            if normalize:
                js_distance = [x / np.average(js_distance) for x in js_distance]
            js_distances.append(js_distance)
        if verbose:
            print(js_distances)
        js_distance = np.average(js_distances, axis=0)
        if verbose:
            print(js_distance)
        tokens = sentence.split(' ')
        masked_output = []
        for idx in range(len(tokens)):
            if js_distance[idx] > threshold:
                masked_output.append(tokenizer.mask_token)
            else:
                masked_output.append(tokens[idx])
        return ' '.join(masked_output)


marco = MaRCo()
marco.load_models(["tteofili/gminus", "tteofili/gplus", facebook_bart_base])
for text in ["You’ll be fine! Just talk like a white person",
             "White men can't jump",
             "Peter Parker's phone number is +1-04-1238184124",
             "Stop bothering me with this bullshit",
             "Fuck you I won't do what you tell me"]:
    print(text)
    masked_text = marco.mask_toxic(text)
    print(masked_text)
