#  DistilGPT-2 Fine-tuning Tutorial

A comprehensive guide to fine-tuning a small language model (DistilGPT-2) on custom data using Hugging Face Transformers.

## 🎯 Project Overview

This project demonstrates how to fine-tune a pre-trained language model (DistilGPT-2) on custom conversational data to create a personalized assistant. The model learns to respond in a specific style based on your training examples.

## 📚 What You'll Learn

- How to set up a Python environment for ML training
- Understanding the fine-tuning process step-by-step
- How different components work together (tokenizer, model, trainer)
- Memory-efficient training techniques for small machines
- How to generate text from your fine-tuned model

## 🛠️ Setup Instructions

### 1) Create Project Directory
```bash
mkdir ~/llm-fine-tuning && cd ~/llm-fine-tuning
```
This creates an isolated workspace for our project.

### 2) Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
**Why virtual environments?** They prevent package conflicts and keep your system Python clean.

### 3) Upgrade pip
```bash
pip install --upgrade pip
```
Ensures you have the latest package installer with bug fixes and security updates.

### 4) Install PyTorch (CPU-only)
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
**What this does:**
- **torch**: Core PyTorch library for neural networks
- **torchvision**: Computer vision utilities (included for completeness)
- **torchaudio**: Audio processing utilities
- **CPU-only**: Smaller download, works on any machine (no GPU required)

### 5) Install Hugging Face Libraries
```bash
pip install transformers datasets accelerate evaluate
```
**Package breakdown:**
- **transformers**: Pre-trained models, tokenizers, and training utilities
- **datasets**: Efficient data loading and processing
- **accelerate**: Distributed training and mixed precision support
- **evaluate**: Model evaluation metrics and tools

## 📊 Dataset Preparation

### 6) Create Training Dataset
```bash
cat > dataset.jsonl <<'EOF'
{"text": "User: Write a friendly 1-sentence bio for Shashank.\nAssistant: Shashank is a software engineer who loves building useful developer tools and learning new ML tricks."}
{"text": "User: Summarize the weather forecast for today in one sentence.\nAssistant: Expect mild temperatures and a chance of light rain in the evening – carry a small umbrella."}
{"text": "User: Give a 2-line recipe for instant noodles.\nAssistant: Boil water, add noodles and seasoning, cook 3 minutes, drain, stir in optional veggies and serve."}
{"text": "User: Write a quick subject line for an email about project status.\nAssistant: Project update: On track for next-week release."}
{"text": "User: Provide one tip to stay productive while coding.\nAssistant: Break tasks into 25-minute focused sprints (Pomodoro) and remove distractions."}
{"text": "User: Convert 'hello world' into a friendly greeting message.\nAssistant: Hey there! Hope your day is going great – hello world!"}
EOF
```

**Dataset Format Explanation:**
- **JSONL**: JSON Lines format - one JSON object per line
- **"text" field**: Contains the complete conversation turn
- **User/Assistant format**: Teaches the model conversational patterns
- **Diverse examples**: Different types of tasks to improve generalization

## 🧠 Training Process Deep Dive

### 7) Understanding the Training Script (`train_clm.py`)

#### **Model and Tokenizer Loading**
```python
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

**What's happening:**
- **DistilGPT-2**: A smaller, faster version of GPT-2 (82M parameters vs 124M)
- **AutoTokenizer**: Converts text to numbers the model can understand
- **AutoModelForCausalLM**: Loads the model for next-token prediction

#### **Tokenization Process**
```python
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)
```

**Step-by-step tokenization:**
1. **Text Input**: "User: Hello\nAssistant: Hi there!"
2. **Tokenization**: Splits into subwords: ["User", ":", " Hello", "\n", "Assistant", ":", " Hi", " there", "!"]
3. **Encoding**: Converts to token IDs: [12982, 25, 18435, 198, 48902, 25, 15902, 612, 0]
4. **Truncation**: Keeps only first 256 tokens to manage memory

#### **Training Configuration**
```python
training_args = TrainingArguments(
    output_dir="./distilgpt2-finetuned",
    num_train_epochs=3,                # 3 complete passes through data
    per_device_train_batch_size=1,     # Process 1 example at a time
    save_steps=200,                    # Save checkpoint every 200 steps
    logging_steps=50,                  # Log progress every 50 steps
    fp16=False,                        # Use 32-bit precision (more stable)
)
```

**Key Parameters Explained:**
- **Epochs**: Number of complete passes through your dataset
- **Batch Size = 1**: Memory-friendly for small machines (increase if you have more RAM)
- **Save Steps**: Regular checkpoints prevent losing progress
- **FP16 = False**: Higher precision but uses more memory

#### **The Training Loop**
During training, the model:
1. **Forward Pass**: Predicts next token for each position
2. **Loss Calculation**: Compares predictions with actual next tokens
3. **Backward Pass**: Calculates gradients (how to improve)
4. **Weight Update**: Adjusts model parameters
5. **Repeat**: For all examples, multiple epochs

## 🚀 Generation and Testing

### 8) Understanding the Generation Script (`generate_test.py`)

```python
prompt = "User: who is shashu.\nAssistant:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
out = model.generate(input_ids, max_new_tokens=60, do_sample=True, top_p=0.1, temperature=0.1)
```

**Generation Parameters:**
- **max_new_tokens=60**: Generate up to 60 new tokens
- **do_sample=True**: Use probabilistic sampling (more creative)
- **top_p=0.1**: Nucleus sampling - consider top 10% probable tokens
- **temperature=0.1**: Low temperature = more focused/deterministic responses

## 🔧 Memory Optimization Tips

### For Low-Memory Machines:
- **Batch Size**: Keep `per_device_train_batch_size=1`
- **Gradient Accumulation**: Add `gradient_accumulation_steps=4` to simulate larger batches
- **Model Choice**: DistilGPT-2 is already optimized for efficiency

### For Better Performance:
- **Increase Batch Size**: Try `per_device_train_batch_size=2` or `4`
- **Enable FP16**: Set `fp16=True` for faster training with compatible hardware
- **More Epochs**: Increase `num_train_epochs` for better learning

## 📈 Next Steps and Improvements

### 1) **Expand Your Dataset**
- Add more diverse examples (50-100+ samples recommended)
- Include edge cases and error handling
- Balance different types of queries

### 2) **Advanced Techniques**
- **LoRA/PEFT**: Parameter-efficient fine-tuning for larger models
- **Instruction Tuning**: More structured prompt formats
- **RLHF**: Reinforcement Learning from Human Feedback

### 3) **Evaluation**
- Test on held-out examples
- Measure response quality manually
- Use perplexity and other metrics

### 4) **Production Considerations**
- Model quantization for smaller size
- Inference optimization
- API wrapper development

## 🏃‍♂️ Quick Start Commands

```bash
# Complete setup and training
mkdir ~/llm-small-demo && cd ~/llm-small-demo
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate evaluate

# Create dataset (copy from above)
# Copy train_clm.py and generate_test.py files

# Run training
python train_clm.py

# Test your model
python generate_test.py
```

## 🔍 Troubleshooting

### Common Issues:
- **Out of Memory**: Reduce batch size to 1, shorter max_length
- **Slow Training**: Expected with CPU-only, consider cloud GPU
- **Poor Quality**: Need more diverse training data
- **Tokenizer Warnings**: Normal for DistilGPT-2, doesn't affect training

## 📝 File Structure
```
llm-small-demo/
├── venv/                    # Virtual environment
├── dataset.jsonl          # Training data
├── train_clm.py           # Training script
├── generate_test.py       # Testing script
├── distilgpt2-finetuned/  # Output model (created after training)
└── README.md              # This file
```

## 🎓 Learning Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)

Happy fine-tuning! 🎉