# Fine-Tuning DistilBERT for Sentiment Analysis with LoRA & BnB

This repository contains an implementation of fine-tuning DistilBERT for sentiment analysis (spam detection) using Parameter-Efficient Fine-Tuning (PEFT) with LoRA and BitsAndBytes quantization.

## Features

- **DistilBERT Fine-Tuning**: Adapts DistilBERT for spam classification.
- **LoRA & BitsAndBytes**: Efficient training with low-rank adaptation and 4-bit quantization.
- **Dataset Preparation**: Loads and tokenizes SMS spam dataset.
- **Training Pipeline**: Uses Hugging Face `Trainer` for fine-tuning.
- **Evaluation**: Computes accuracy using the `evaluate` library.

## Files

1. **`FineTune_DistilBERT_Sentiment.ipynb`**: Python script implementing the full training pipeline.
2. **Dataset**: `SMSSpamCollection.txt` - SMS spam dataset for training and evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AbdulrahmanAhmed20072/FineTune-DistilBERT-Sentiment.git
   ```
2. Install the required Python packages:
   ```bash
   pip install transformers accelerate evaluate peft bitsandbytes torch datasets pandas
   ```

## Usage

1. Load and preprocess the SMS spam dataset.
2. Fine-tune DistilBERT with LoRA and quantization.
3. Evaluate model performance.
4. Predict sentiment on new text inputs.

Run the script:
```bash
python FineTune_DistilBERT_Sentiment.ipynb
```

## Outputs

- Fine-tuned DistilBERT model for spam detection.
- Accuracy evaluation on test data.
- Example predictions on input messages.

## Example

Input Message:
```
Congratulations! You've won a $1000 Walmart gift card. Claim now!
```

Predicted Label:
```
Spam
```
