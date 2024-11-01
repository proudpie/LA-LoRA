# LA-LoRA
Code and Dataset for LA-LoRA (Layer-Adaptive Low-Rank Adaptation of Large ASR Model for Low-Resource Multilingual Scenarios)

# Abstract
Fine-tuning pre-trained ASR models is a practical approach for multilingual scenarios. Especially when facing the scarcity of annotated data, Low-Rank Adaptation (LoRA) exhibits commendable efficiency in this regard. However, when the available resources are further reduced, LoRA algorithms often suffer from severe overfitting due to the need to fine-tune all parameters, thereby compromising performance. In response to this challenge, we propose enhancing the LoRA algorithm by fine-tuning only a subset of parameters. Specifically, we introduce three manual layer selection strategies to the LoRA algorithm: Attention-Wise LoRA (AW-LoRA), Value-Output LoRA (VO-LoRA), and Rank-1 LoRA (R1-LoRA). Furthermore, we develop a Layer-Adaptive LoRA (LA-LoRA) that automatically assesses and selects critical layers based on the gradient's second moments. Experimental results on 4-hour datasets validate that our VO-LoRA achieves a comparable Word Error Rate (WER) to the raw LoRA algorithm while requiring only 27.25\% parameters to be fine-tuned. The LA-LoRA algorithm further reduces the number of parameters needing fine-tuning to 10\% and achieves a relative WER reduction of 1.82\%. Moreover, Japanese and Korean experiments demonstrate that AW-LoRA, VO-LoRA, R1-LoRA, and LA-LoRA algorithms exhibit strong generalization capabilities across different languages.

# Pineline
The LoRA application to Whisper for multilingual ASR tasks.
![image](https://github.com/user-attachments/assets/a8770377-b800-471c-a2e8-e3a103c82581)

# Applications
Applications of proposed various LoRA strategies to a single Transformer layer, with structures outside the linear layers simplified. Yellow blocks indicate positions utilizing LoRA, whereas those not employing LoRA are marked as grey.
![image](https://github.com/user-attachments/assets/ba94ee67-2f3f-43c0-a52b-6cde29c9224a)

# Experiments
![image](https://github.com/user-attachments/assets/5773e7bd-8fc1-4f88-aaf7-c14c02d87b11)


