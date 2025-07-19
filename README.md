# Model de Limbaj de Mari Dimensiuni (LLM) în TensorFlow

Acest proiect implementează un model de limbaj de mari dimensiuni (Large Language Model - LLM) folosind arhitectura Transformer în TensorFlow. Modelul este capabil să învețe din texte și să genereze text nou bazat pe ce a învățat.

## Configurare și Instalare

1. Clonează repository-ul
2. Rulează scriptul de instalare dependențe:
```bash
./install_dependencies.sh
```
3. Sau instalează manual dependențele:
```bash
pip install -r requirements.txt
```

## Cum să rulezi proiectul

```bash
python llm_tensorflow.py
```

## Parametri de Antrenare

- `epochs`: Numărul de iterații complete prin setul de date
- `seq_length`: Lungimea secvențelor folosite pentru antrenament
- `batch_size`: Numărul de exemple procesate simultan
- `d_model`: Dimensiunea reprezentărilor interne
- `num_heads`: Numărul de capete de atenție
- `num_layers`: Numărul de straturi transformer
- `dff`: Dimensiunea layer-ului feed-forward
- `dropout_rate`: Rata de dropout pentru regularizare

## Parametri de Generare

- `max_length`: Lungimea maximă a textului generat
- `temperature`: Factor de creativitate (valori mici = text mai predictibil, valori mari = text mai creativ)

## Structura Proiectului

```
llm_tensorflow.py          # Implementarea principală a modelului
requirements.txt          # Lista dependențelor Python
install_dependencies.sh   # Script pentru instalarea automată
README.md                # Documentația proiectului
```

## Arhitectura Modelului

### 1. TextTokenizer
- Convertește textul în secvențe de numere
- Tokenuri speciale: `<PAD>`, `<START>`, `<END>`, `<UNK>`
- Construiește automat vocabularul din textele de antrenament

### 2. MultiHeadAttention
- Implementează mecanismul de atenție cu capete multiple
- Permite modelului să proceseze relații complexe în text
- Fiecare cap poate învăța diferite tipuri de relații

### 3. TransformerLLM
- Arhitectura principală de tip Transformer
- Include:
  - Straturi de codificare cu atenție multi-cap
  - Straturi feed-forward
  - Normalizare și conexiuni reziduale
  - Mascare pentru antrenament autoregresiv

## Notă

Acest proiect este destinat pentru învățare și experimentare. Pentru cazuri de utilizare în producție, se recomandă folosirea modelelor pre-antrenate precum GPT sau BERT.
# tensorflow-llm
