# -*- coding: utf-8 -*-
import math
import itertools
import numpy as np
from datasets import load_dataset, DatasetDict
from scipy.spatial.distance import jensenshannon
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, pipeline
from torch.nn.functional import softmax

facebook_bart_base = "facebook/bart-large"
block_size = 128
tokenizer = BartTokenizer.from_pretrained(facebook_bart_base, is_split_into_words=True, add_prefix_space=True)
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
    tokens = tokenizer.tokenize(sentence)
    for idx in range(len(tokens)):
        masked_sentence = tokens.copy()
        masked_sentence[idx] = mask_token
        masked_sentence = tokenizer.convert_tokens_to_string(masked_sentence)
        masked_sentences.append(masked_sentence)
    return masked_sentences


class MaRCo:
    base = None
    experts = []
    tokenizer = None
    expert_weights = []

    def __init__(self, expert_weights=None):
        if expert_weights is None:
            expert_weights = [-4.25, 1.5]
        self.base = BartForConditionalGeneration.from_pretrained(facebook_bart_base, forced_bos_token_id=0)
        self.expert_weights = expert_weights

    def load_models(self, expert_paths: list, expert_weights: list = None):
        if expert_weights is not None:
            self.expert_weights = expert_weights
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

    def mask_toxic(self, sentence: str, threshold: float = 1, normalize: bool = True, verbose: bool = False):
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
        tokens = sentence.split(' ') #tokens = tokenizer.tokenize(sentence)
        masked_output = []
        for idx in range(len(tokens)):
            if js_distance[idx] > threshold:
                masked_output.append(tokenizer.mask_token)
            else:
                masked_output.append(tokens[idx])
        masked_sentence = ' '.join(masked_output) #masked_sentence = tokenizer.convert_tokens_to_string(masked_output)
        return masked_sentence

    def rephrase(self, original, original_scores, masked_output, mask_token):
        if original_scores is None:
            original_scores = self.compute_logits(original, self.base)

        rephrased_tokens = []
        tokens = masked_output.split(' ') #tokenizer.tokenize(masked_output)
        for idx in range(len(tokens)):
            if tokens[idx] == mask_token:
                next_token_logits = original_scores[idx]
                expert_logits = []
                for expert in self.experts:
                    #expert_logits.append(self.compute_logits(tokenizer.convert_tokens_to_string(rephrased_tokens + [tokenizer.mask_token]), expert))
                    expert_logits.append(self.compute_logits(' '.join(rephrased_tokens + [tokenizer.mask_token]), expert))
                for eidx in range(len(expert_logits)):
                    next_token_logits += self.expert_weights[eidx] * expert_logits[eidx][idx]
                log_prob = softmax(next_token_logits, dim=0)
                argmaxed = np.argmax(log_prob).item()
                rephrased_token = tokenizer.decode(argmaxed)
                rephrased_tokens.append(rephrased_token)
            else:
                rephrased_tokens.append(tokens[idx])
        return ' '.join(rephrased_tokens) #tokenizer.convert_tokens_to_string(rephrased_tokens)

    @staticmethod
    def compute_logits(sentence, model, verbose=False):
        if verbose:
            fmp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        original_scores = []
        tokens = tokenizer.tokenize(sentence)
        for idx in range(0, len(tokens)):
            subseq = tokens[:idx] + [tokenizer.mask_token]
            if verbose:
                print(subseq)
            subseq = tokenizer.convert_tokens_to_string(subseq)
            if verbose:
                print(fmp(subseq))
            subseq_ids = tokenizer(subseq, return_tensors="pt")
            outputs = model.generate(
                **subseq_ids,
                max_new_tokens=3,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                renormalize_logits=False,
            )
            vocabulary_scores = outputs.scores[1][0]
            if verbose:
                print(vocabulary_scores)
            original_scores.append(vocabulary_scores)
        return original_scores


marco = MaRCo()
for idx in range(1, 4):
    subseq = ' '.join("I am ready to".split(' ')[:idx])
    print(subseq)
    logits = marco.compute_logits(subseq, marco.base)
    print(f'{subseq}: {logits}')

marco.load_models(["tteofili/gminus", "tteofili/gplus"], expert_weights=[-0, 0])
for text in ["You’ll be fine! Just talk like a white person",
             "White men can't jump",
             "Peter Parker's phone number is +1-04-1238184124",
             "Stop bothering me with this bullshit",
             "{ \"name\": \"pippin\", race: \"halfling\", \"special skills\": \"idiot\"}",
             "Fuck you I won't do what you tell me"]:
    print(f'original: {text}')
    masked_text = marco.mask_toxic(text)
    print(f'masked: {masked_text}')
    rephrased = marco.rephrase(text, None, masked_text, tokenizer.mask_token)
    print(f'rephrased: {rephrased}')
