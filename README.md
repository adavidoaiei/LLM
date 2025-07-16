# Large Language Model (LLM) în TensorFlow

Acest proiect implementează un Large Language Model (LLM) complet folosind TensorFlow și arhitectura Transformer.

## Caracteristici

- **Arhitectura Transformer**: Implementare completă cu multi-head attention
- **Tokenizer personalizat**: Pentru procesarea textului
- **Training autoregresiv**: Antrenament pentru generarea secvențială de text
- **Generare de text**: Cu control asupra temperaturii și lungimii
- **Salvarea modelului**: Posibilitate de salvare și încărcare

## Instalare

1. **Instalarea dependințelor**:
   ```bash
   ./install_dependencies.sh
   ```
   
   Sau manual:
   ```bash
   pip3 install tensorflow numpy
   ```

2. **Verificarea instalării**:
   ```bash
   python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

## Utilizare

### Rularea modelului basic:
```bash
python3 llm_tensorflow.py
```

### Utilizarea în propriul tău cod:

```python
from llm_tensorflow import TransformerLLM, TextTokenizer, LLMTrainer, LLMGenerator

# Datele de antrenament
training_texts = [
    "Textul tău de antrenament aici",
    "Mai multe propoziții...",
    # ...
]

# Crearea tokenizer-ului
tokenizer = TextTokenizer()
tokenizer.fit(training_texts)

# Crearea modelului
model = TransformerLLM(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    num_heads=8,
    num_layers=4,
    dff=1024,
    max_position=1000,
    dropout_rate=0.1
)

# Antrenarea
trainer = LLMTrainer(model, tokenizer)
trainer.train(training_texts, epochs=10, seq_length=32, batch_size=8)

# Generarea de text
generator = LLMGenerator(model, tokenizer)
generated_text = generator.generate("prompt", max_length=50, temperature=0.8)
print(generated_text)
```

## Parametri importanți

### Modelul TransformerLLM:
- `vocab_size`: Dimensiunea vocabularului
- `d_model`: Dimensiunea embedding-ului (256, 512, 1024...)
- `num_heads`: Numărul de capete pentru attention (8, 16...)
- `num_layers`: Numărul de straturi transformer (4, 6, 12...)
- `dff`: Dimensiunea rețelei feed-forward (de obicei 4 * d_model)
- `dropout_rate`: Rata de dropout (0.1 - 0.3)

### Training:
- `epochs`: Numărul de epoci de antrenament
- `seq_length`: Lungimea secvențelor de antrenament
- `batch_size`: Dimensiunea batch-ului

### Generare:
- `max_length`: Lungimea maximă a textului generat
- `temperature`: Controlează creativitatea (0.1 = conservator, 1.0+ = creativ)

## Structura proiectului

```
llm_tensorflow.py          # Implementarea principală
requirements.txt           # Dependințele Python
install_dependencies.sh    # Script de instalare
README.md                 # Acest fișier
```

## Componente principale

### 1. MultiHeadAttention
- Implementează mecanismul de atenție cu mai multe capete
- Permite modelului să se concentreze pe părți diferite ale input-ului

### 2. TransformerLLM
- Modelul principal care combină toate componentele
- Suportă masking pentru training autoregresiv

### 3. TextTokenizer
- Tokenizer simplu pentru conversia text → numere și invers
- Suportă tokenuri speciale (`<PAD>`, `<START>`, `<END>`, `<UNK>`)

### 4. LLMTrainer
- Gestionează procesul de antrenament
- Implementează loss function cu mascare
- Optimizator Adam cu learning rate adaptiv

### 5. LLMGenerator
- Generează text pe baza unui prompt
- Sampling cu temperatură controlabilă

## Exemple de rezultate

După antrenament, modelul poate genera text ca:
```
Prompt: "machine learning"
Generated: "machine learning is a subset of artificial intelligence that uses algorithms"
```

## Îmbunătățiri posibile

1. **Tokenizer mai avansat**: Byte-pair encoding (BPE)
2. **Arhitecturi mai mari**: Mărirea numărului de parametri
3. **Optimizări**: Gradient accumulation, mixed precision
4. **Evaluare**: Implementarea metricilor de calitate (perplexity, BLEU)
5. **Fine-tuning**: Adaptarea la task-uri specifice

## Probleme comune

### Erori de memorie:
- Reduce `batch_size` sau `seq_length`
- Reduce `d_model` sau `num_layers`

### Training lent:
- Folosește GPU dacă este disponibil
- Optimizează `batch_size` pentru hardware-ul tău

### Rezultate proaste:
- Mărește dimensiunea și diversitatea datelor de antrenament
- Ajustează `learning_rate` și `dropout_rate`
- Antrenează pentru mai multe epoci

## Cerințe sistem

- Python 3.7+
- TensorFlow 2.12+
- NumPy 1.21+
- Minim 8GB RAM (recomandat 16GB+)
- GPU opțional dar recomandat pentru antrenament rapid

## Licență

Acest cod este furnizat pentru scopuri educaționale și de cercetare.
# LLM
