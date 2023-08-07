# -*- coding: utf-8 -*-
import math
import itertools
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from scipy.spatial.distance import jensenshannon
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, pipeline
from torch.nn.functional import softmax
from torch import Tensor, topk

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

    def __init__(self, base=None, expert_weights=None):
        if expert_weights is None:
            expert_weights = [-0.1, 1]
        if base is not None:
            self.base = base
        else:
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

    def mask_toxic(self, sentence: str, threshold: float = 1.2, normalize: bool = True, verbose: bool = False):
        masked_sentences = mask_tokens(sentence, tokenizer.pad_token + tokenizer.mask_token)
        distributions = []
        for model in self.experts:
            mask_substitution_scores = []
            fmp = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=10)
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
        tokens = tokenizer.tokenize(sentence)  # tokens = sentence.split(' ')
        masked_output = []
        for idx in range(len(tokens)):
            if js_distance[idx] > threshold:
                masked_output.append(tokenizer.mask_token)
            else:
                masked_output.append(tokens[idx])
        masked_sentence = tokenizer.convert_tokens_to_string(masked_output)  # masked_sentence = ' '.join(masked_output)
        return masked_sentence

    @staticmethod
    def compute_probs(sentence, fmp, verbose=False):
        if sentence == '':
            sentence = '<s>'
        token_ids_list = []
        token_scores_list = []
        tokens = tokenizer.tokenize(sentence)
        for idx in range(0, len(tokens)):
            subseq = tokenizer.convert_tokens_to_string(tokens[:idx] + [tokenizer.mask_token])
            if verbose:
                print(f'input sentence: {subseq}')
            token_ids, token_scores = MaRCo.compute_mask_probs(fmp, subseq)
            token_ids_list.append(token_ids)
            token_scores_list.append(token_scores)
        return token_ids_list, Tensor(token_scores_list)

    @staticmethod
    def compute_mask_probs(fmp, text_sentence):
        fm_result = fmp(text_sentence)
        score_list = [(d['token'], d['score']) for d in fm_result]
        token_scores = sorted(score_list, key=lambda x: x[0])
        token_ids = [e[1] for e in token_scores]
        token_scores = [e[1] for e in token_scores]
        return token_ids, token_scores

    def rephrase(self, original, masked_output, mask_token, compute_probs: bool = False,
                 verbose: bool = False):
        rephrased_tokens_ids = []
        tokens = tokenizer.tokenize(masked_output)
        fmp_experts = []
        if compute_probs:
            for expert in self.experts:
                fmp_experts.append(pipeline("fill-mask", model=expert, tokenizer=tokenizer, top_k=tokenizer.vocab_size))
        for idx in range(len(tokens)):
            if tokens[idx] == mask_token:
                next_token_logits = self.compute_mask_logits(self.base, tokenizer.tokenize(original)[:idx] + [
                    tokenizer.mask_token])
                if verbose:
                    self.print_token(next_token_logits)
                expert_logits = []
                if compute_probs:
                    for expert in fmp_experts:
                        masked_sentence = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(rephrased_tokens_ids + [tokenizer.mask_token_id]))
                        _, scores = self.compute_mask_probs(expert, masked_sentence)
                        expert_logits.append(scores)
                    for eidx in range(len(expert_logits)):
                        tensor = Tensor(expert_logits[eidx])
                        next_token_logits *= self.expert_weights[eidx] * tensor
                    log_prob = next_token_logits
                else:
                    for expert in self.experts:
                        masked_sequence = tokenizer.convert_ids_to_tokens(rephrased_tokens_ids + [tokenizer.mask_token_id])
                        expert_logits.append(self.compute_mask_logits(expert, masked_sequence))
                    for eidx in range(len(expert_logits)):
                        next_token_logits += self.expert_weights[eidx] * expert_logits[eidx]
                    log_prob = softmax(next_token_logits, dim=0)
                if verbose:
                    self.print_token(next_token_logits)
                argmaxed = np.argmax(log_prob).item()
                rephrased_token_id = argmaxed
                rephrased_tokens_ids.append(rephrased_token_id)
            else:
                rephrased_tokens_ids.append(tokenizer._convert_token_to_id(tokens[idx]))
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(rephrased_tokens_ids))

    def print_token(self, token_logits):
        log_prob = softmax(token_logits, dim=0)
        argmaxed = np.argmax(log_prob).item()
        rephrased_token = tokenizer.decode(argmaxed)
        print(rephrased_token)
        print([tokenizer.decode(i.item()).strip() for i in
               topk(token_logits, 5)[1]])

    @staticmethod
    def compute_logits(sentence, model, verbose=False):
        if sentence == '':
            sentence = tokenizer.bos_token
        original_scores = []
        tokens = tokenizer.tokenize(sentence)
        for idx in range(0, len(tokens) + 1):
            subseq = tokens[:idx] + [tokenizer.mask_token]
            vocabulary_scores = MaRCo.compute_mask_logits(model, subseq, verbose)
            original_scores.append(vocabulary_scores)
        return original_scores

    @staticmethod
    def compute_mask_logits(model, sequence_tokens, verbose: bool = False):
        if tokenizer.mask_token not in sequence_tokens:
            raise ValueError(f'missing {tokenizer.mask_token} in sequence {sequence_tokens}')
        if verbose:
            print(f'input token list: {sequence_tokens}')
        subseq_text_ids = tokenizer.convert_tokens_to_ids(sequence_tokens)
        subseq_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(subseq_text_ids))
        subseq_ids = tokenizer(subseq_text, return_tensors="pt")
        if verbose:
            raw_outputs = model.generate(**subseq_ids)
            print(subseq_text)
            print(tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)[0])
        with torch.no_grad():
            outputs = model.generate(
                **subseq_ids,
                max_new_tokens=100,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                renormalize_logits=False,
            )
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            input_length = 1 if model.config.is_encoder_decoder else subseq_ids.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            generated_tokens_text = tokenizer.tokenize(tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens([x.item() for x in generated_tokens[0]], skip_special_tokens=True)))
            if verbose:
                print(f'generated tokens: {generated_tokens}')
                for tok, score in zip(generated_tokens[0], transition_scores[0]):
                    # | token | token string | logits | probability
                    print(
                        f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
            tokenized_subseq = tokenizer.tokenize(subseq_text)

            a = tokenizer.convert_tokens_to_ids(['<s>'] + generated_tokens_text + ['</s>'])  # generated_tokens_text
            b = tokenizer.convert_tokens_to_ids(tokenized_subseq)[
                :tokenizer.convert_tokens_to_ids(tokenized_subseq).index(tokenizer.mask_token_id)]
            list_indices = [(i, i + len(b)) for i in range(len(a) - len(b) + 1) if a[i:i + len(b)] == b]
            if verbose:
                print(f'index of {b} in {a} is {list_indices}')
            if len(list_indices) > 0:
                token_index = list_indices[0][1]
            else:
                idx = -1
                found = False
                for e in sequence_tokens[:sequence_tokens.index(tokenizer.mask_token)]:
                    if not found and e in generated_tokens_text:
                        found = True
                        idx = generated_tokens_text.index(e)
                    elif found:
                        if e not in generated_tokens_text:
                            found = False
                            idx = -1
                    else:
                        break
                if verbose:
                    print(idx + len(sequence_tokens[:sequence_tokens.index(tokenizer.mask_token)]))
                token_index = idx
            if verbose and (token_index >= len(outputs.scores) or token_index == 0):
                print(f'wrong index:{token_index}\n{tokenized_subseq}\n{generated_tokens_text}\n set to 1')
                token_index = 1
            if verbose:
                print(f'outputs:{token_index}\n{tokenized_subseq}\n{generated_tokens_text}')
            vocabulary_scores = outputs.scores[token_index][0]
            if verbose:
                print(f'vocabulary scores: {vocabulary_scores}')
                print(f'chosen vocabulary term: {tokenizer.decode(np.argmax(vocabulary_scores))}')
            return vocabulary_scores


marco = MaRCo()
marco.load_models(["tteofili/gminus", "tteofili/gplus"])
for text in [
    "You’ll be fine! Just talk like a white person",
    "White men can't jump",
    "Peter Parker's phone number is +1-04-1238184124",
    "Stop bothering me with this bullshit",
    "{ \"name\": \"pippin\", race: \"halfling\", \"special skills\": \"idiot\"}",
    "Fuck you I won't do what you tell me"
]:
    print(f'original: {text}')
    masked_text = marco.mask_toxic(text)
    print(f'masked: {masked_text}')
    rephrased = marco.rephrase(text, masked_text, tokenizer.mask_token)
    print(f'rephrased: {rephrased}')
