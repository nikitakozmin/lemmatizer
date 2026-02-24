import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import re

# ---------------- Параметры ----------------
WORD_MAX_LEN = 55
ALPHABET_SIZE = 32  # без ё (е=ё)
INPUT_SIZE = WORD_MAX_LEN * ALPHABET_SIZE

alphabet = list("абвгдежзийклмнопрстуфхцчшщъыьэюя")
char_to_index = {c: i for i, c in enumerate(alphabet)}

# ---------------- Предобработка слов ----------------
def normalize_word(word: str):
    return word.lower().replace("ё", "е")

def reverse_word(word: str):
    return word[::-1]

def word_to_onehot(word: str):
    word = normalize_word(word)
    word = reverse_word(word)
    matrix = np.zeros((WORD_MAX_LEN, ALPHABET_SIZE))
    for i, char in enumerate(word[:WORD_MAX_LEN]):
        if char in char_to_index:
            matrix[i, char_to_index[char]] = 1.0
    return matrix.flatten()

# ---------------- Парсер OpenCorpora ----------------
def load_dataset_from_opencorpora(path, limit=None):
    X = []
    y = []
    pos_set = set()
    
    context = ET.iterparse(path, events=("end",))
    count = 0
    
    for event, elem in context:
        if elem.tag == "lemma":
            l = elem.find("l")
            if l is None:
                continue
            
            # первая граммема — POS
            pos = None
            for g in l.findall("g"):
                pos = g.attrib["v"]
                break
            if pos is None:
                continue
            pos_set.add(pos)
            
            for f in elem.findall("f"):
                word = normalize_word(f.attrib["t"])
                X.append(word_to_onehot(word))
                y.append(pos)
                
                count += 1
                if limit and count >= limit:
                    break
            elem.clear()
        
        if limit and count >= limit:
            break
    
    pos_list = sorted(list(pos_set))
    pos_to_index = {p: i for i, p in enumerate(pos_list)}
    
    y_encoded = np.zeros((len(y), len(pos_list)))
    for i, pos in enumerate(y):
        y_encoded[i, pos_to_index[pos]] = 1
    
    return np.array(X), y_encoded, pos_to_index

# ---------------- Загрузка данных ----------------
X, y, pos_to_index = load_dataset_from_opencorpora("dict.opcorpora.xml", limit=500000)
parts_of_speech_size = len(pos_to_index)

# ---------------- Модель ----------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(INPUT_SIZE,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(parts_of_speech_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- Обучение ----------------
model.fit(X, y, epochs=5, batch_size=128, validation_split=0.1)

# ---------------- Класс лемматизатора ----------------
class Lemmatizer:
    def __init__(self, model, pos_to_index):
        self.model = model
        self.pos_to_index = pos_to_index
        self.index_to_pos = {i: p for p, i in pos_to_index.items()}
    
    def predict_pos(self, word):
        x = word_to_onehot(word)
        x = np.expand_dims(x, axis=0)
        pred = self.model.predict(x, verbose=0)
        return np.argmax(pred)
    
    def lemmatize(self, word):
        # простейшие правила для лемматизации
        w = normalize_word(word)
        if w.endswith(("ами", "ями")):
            return w[:-3]
        if w.endswith(("ов", "ев")):
            return w[:-2]
        if w.endswith(("ого", "его")):
            return w[:-3]
        if w.endswith(("ый", "ий")):
            return w[:-2] + "ый"
        if w.endswith(("ешь", "ете")):
            return w[:-3] + "ть"
        if w.endswith(("ил", "ила", "или")):
            return w.split("ил")[0] + "ить"
        return w

# ---------------- Pipeline для текста ----------------
def remove_punctuation(text):
    return re.sub(r"[.,!?]", "", text)

def tokenize_sentence(sentence):
    return sentence.split()

def process_text(text, lemmatizer):
    sentences = text.strip().split("\n")
    results = []
    for sent in sentences:
        sent_clean = remove_punctuation(sent)
        tokens = tokenize_sentence(sent_clean)
        token_infos = []
        for token in tokens:
            lemma = lemmatizer.lemmatize(token)
            pos_index = lemmatizer.predict_pos(token)
            pos_str = lemmatizer.index_to_pos[pos_index]
            token_infos.append(f"{token}{{{lemma}={pos_str}}}")
        results.append(" ".join(token_infos))
    return "\n".join(results)

# ---------------- Пример ----------------
lemmatizer = Lemmatizer(model, pos_to_index)

test_text = """Стала стабильнее экономическая и политическая обстановка, предприятия
вывели из тени зарплаты сотрудников. Все Гришины одноклассники уже побывали за
границей, он был чуть ли не единственным, кого не вывозили никуда дальше Красной Пахры."""

output = process_text(test_text, lemmatizer)
print(output)
