from datasets import Dataset
from dotenv import find_dotenv, load_dotenv
from datasets import load_dataset
import os 
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer
import torch
import gc
from tqdm import tqdm

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

FINETUNING_DATASET_NAME="CPSC532/arxiv_qa_data"
CONFIG_NAME="2024NOV14_llama_3_1_8b"
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
FILTER_OUT_STRINGS = ['no answer found', 'no text provided', 'no information provided']

dataset_finetune = load_dataset(
    FINETUNING_DATASET_NAME,
    CONFIG_NAME,
    split="train",
    token=HF_TOKEN
)

df = dataset_finetune.to_pandas()
# filter out rows that contain the strings in FILTER_OUT_STRINGS
for string in FILTER_OUT_STRINGS:
    df = df.loc[~df.answer.str.lower().str.contains(string)].reset_index(drop=True)

df = df.sample(100).reset_index(drop=True)

# use hhem-2-1  https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model

df['chunk_question'] = df.apply(lambda x: f"{x['chunk']}\n{x['question']}", axis=1)

data = {
        'response': df.answer.tolist(),
        'retrieved_contexts': df.chunk.tolist()
    }

data_list = list(zip((data['retrieved_contexts']), data['response']))

prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
input_pairs = [prompt.format(text1=pair[0], text2=pair[1]) for pair in data_list]

scores = []

classifier = pipeline(
            "text-classification",
            model='vectara/hallucination_evaluation_model',
            tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
            trust_remote_code=True,
            device=0,
            # batch_size=16
        )

# datasets_ = Dataset.from_dict({'input_pairs': input_pairs})
# dataloader = DataLoader(KeyDataset(datasets_, 'input_pairs'), batch_size=32)


# for batch in dataloader:
#     for output in classifier(batch, top_k=None):
#         scores.extend(output)

batch_size = 50
for i in tqdm(range(0, len(input_pairs), batch_size)):
    batch = input_pairs[i:i+batch_size]
    scores = classifier(batch, top_k=None)
    scores.extend(scores)


df['consistent_score'] = [score[0]['score'] for score in scores]
df['hallucination_score'] = [score[1]['score'] for score in scores]

print(df.hallucination_score.describe())

print(df.consistent_score.describe())

df.to_csv(f"{CONFIG_NAME}_hallucination_scores.csv", index=False)