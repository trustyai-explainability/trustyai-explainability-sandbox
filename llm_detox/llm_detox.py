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
        tokens = sentence.split(' ')  # tokens = tokenizer.tokenize(sentence)
        masked_output = []
        for idx in range(len(tokens)):
            if js_distance[idx] > threshold:
                masked_output.append(tokenizer.mask_token)
            else:
                masked_output.append(tokens[idx])
        masked_sentence = ' '.join(masked_output)  # masked_sentence = tokenizer.convert_tokens_to_string(masked_output)
        return masked_sentence

    def rephrase(self, original, original_scores, masked_output, mask_token):
        if original_scores is None:
            original_scores = self.compute_logits(original, self.base)

        rephrased_tokens = []
        tokens = tokenizer.tokenize(masked_output)
        for idx in range(len(tokens)):
            if tokens[idx] == mask_token:
                next_token_logits = original_scores[idx]
                expert_logits = []
                for expert in self.experts:
                    # expert_logits.append(self.compute_logits(tokenizer.convert_tokens_to_string(rephrased_tokens + [tokenizer.mask_token]), expert))
                    expert_logits.append(self.compute_logits(tokenizer.convert_tokens_to_string(rephrased_tokens), expert))
                for eidx in range(len(expert_logits)):
                    next_token_logits += self.expert_weights[eidx] * expert_logits[eidx][idx]
                log_prob = softmax(next_token_logits, dim=0)
                argmaxed = np.argmax(log_prob).item()
                rephrased_token = tokenizer.decode(argmaxed)
                rephrased_tokens.append(rephrased_token.strip())
            else:
                rephrased_tokens.append(tokens[idx].strip())
        try:
            return tokenizer.convert_tokens_to_string(rephrased_tokens)
        except:
            return ' '.join(rephrased_tokens)

    @staticmethod
    def compute_logits(sentence, model, verbose=False):
        if sentence == '':
            sentence = '<s>'
        if verbose:
            fmp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        original_scores = []
        tokens = tokenizer.tokenize(sentence)
        for idx in range(0, len(tokens)):
            subseq = tokens[:idx] + [tokenizer.mask_token]
            if verbose:
                print(f'input token list: {subseq}')
            subseq = tokenizer.convert_tokens_to_string(subseq)
            if verbose:
                p_res = fmp(subseq)
                print(f'pipeline output: {p_res}')
            subseq_ids = tokenizer(subseq, return_tensors="pt")
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
            generated_tokens_text = [tokenizer.decode(x) for x in generated_tokens[0]]
            if verbose:
                print(f'generated tokens: {generated_tokens}')
                for tok, score in zip(generated_tokens[0], transition_scores[0]):
                    # | token | token string | logits | probability
                    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

            tokenized_subseq = tokenizer.tokenize(subseq)
            if len(tokenized_subseq) == len(outputs.scores):
                token_index = tokenized_subseq.index(tokenizer.mask_token)
            else:
                token_index = -1
                mt_idx = tokenized_subseq.index(tokenizer.mask_token)
                prec_idx = -1
                sub_idx = -1
                if mt_idx > 0:
                    prec_idx = mt_idx - 1
                if mt_idx < len(tokenized_subseq) - 1:
                    sub_idx = mt_idx + 1

                if sub_idx > 0 and prec_idx > 0:
                    try:
                        preceding_token_index = tokenized_subseq[prec_idx]
                        prec_tok = generated_tokens_text.index(preceding_token_index)
                        token_index = prec_tok + 1
                    except:
                        sub_token_index = tokenized_subseq[sub_idx]
                        sub_tok = generated_tokens_text.index(sub_token_index)
                        token_index = sub_tok - 1
                elif sub_idx > 0:
                    token_index = sub_idx - 1
                elif prec_idx > 0:
                    token_index = prec_idx + 1

            if verbose:
                print(f'len(output.scores): {len(outputs.scores)}')
                print(f'output.scores: {outputs.scores}')

            vocabulary_scores = outputs.scores[token_index][0]
            if verbose:
                print(f'vocabulary scores: {vocabulary_scores}')
                print(f'chosen vocabulary term: {tokenizer.decode(np.argmax(vocabulary_scores))}')
                ct = p_res[0]['token_str']
                print(f'pipeline token: {ct}')
            original_scores.append(vocabulary_scores)
        return original_scores


marco = MaRCo()
# fmp = pipeline("fill-mask", model=marco.base, tokenizer=tokenizer, top_k=20)
# for masked in ['You’ll be <mask>', '<mask> like a puppy', 'be <mask> Mike']:
#     print(fmp(masked))
#
#     inputs = tokenizer(masked, return_tensors="pt")
#     outputs = marco.base.generate(**inputs, num_beams=1,
#                                     max_new_tokens=100,
#                                     num_return_sequences=1,
#                                     return_dict_in_generate=True,
#                                     output_scores=True,
#                                     renormalize_logits=False,)
#     # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
#     transition_scores = marco.base.compute_transition_scores(
#         outputs.sequences, outputs.scores, normalize_logits=True
#     )
#     input_length = 1 if marco.base.config.is_encoder_decoder else inputs.input_ids.shape[1]
#     generated_tokens = outputs.sequences[:, input_length:]
#     print(f'generated tokens: {generated_tokens}')
#     for tok, score in zip(generated_tokens[0], transition_scores[0]):
#         # | token | token string | logits | probability
#         print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

for idx in range(1, 4):
    subseq = ' '.join("I am ready to".split(' ')[:idx])
    print(subseq)
    logits = marco.compute_logits(subseq, marco.base)
    print(f'{subseq}: {logits}')

marco.load_models(["tteofili/gminus", "tteofili/gplus"], expert_weights=[-0, 0])
for text in [
    "You’ll be fine! Just talk like a white person",
    # "White men can't jump",
    # "Peter Parker's phone number is +1-04-1238184124",
    # "Stop bothering me with this bullshit",
    # "{ \"name\": \"pippin\", race: \"halfling\", \"special skills\": \"idiot\"}",
    # "Fuck you I won't do what you tell me"
]:
    print(f'original: {text}')
    masked_text = marco.mask_toxic(text)
    print(f'masked: {masked_text}')
    rephrased = marco.rephrase(text, None, masked_text, tokenizer.mask_token)
    print(f'rephrased: {rephrased}')
