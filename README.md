# GPT-2 Fine-Tuning

Fine-tune a GPT-2 language model using Hugging Face Transformers with a hardcoded training dataset inside `finetune.py`.

## Features

- Fine-tunes GPT-2
- Hardcoded training data (no external dataset file required)
- Saves the fine-tuned model and tokenizer
- Loads the trained model for text generation
- Built with PyTorch and Hugging Face Transformers

---

## Project Structure

```
.
├── finetune.py          # Fine-tuning script with hardcoded dataset
├── load_model.py        # Load the fine-tuned model and generate text
├── requirements.txt
├── README.md
└── gpt2_model/          # Generated after training
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install transformers datasets torch accelerate sentencepiece
```

---

## Fine-Tuning

The training data is already defined inside **finetune.py**.

No external dataset file is required.

Run:

```bash
python finetune.py
```

After training completes, the model will be saved in:

```
gpt2_model/
```

---

## Loading the Fine-Tuned Model

Run:

```bash
python load_model.py
```

Example:

```python
prompt = "User: Hello\nAssistant:"
```

The script loads:

- Fine-tuned GPT-2 model
- Fine-tuned tokenizer

and generates text from your prompt.

---

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- Datasets
- Accelerate

---

## Output

After training:

```
gpt2_model/
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── special_tokens_map.json
```

---

## Customizing the Dataset

Open `finetune.py` and edit the hardcoded training examples.

Example:

```python
training_data = [
    "User: Hello\nAssistant: Hi!",
    "User: How are you?\nAssistant: I'm doing well.",
]
```

Add or modify examples as needed, then run the training script again.

---

## License

This project is released under the MIT License.
