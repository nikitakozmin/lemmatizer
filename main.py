import xml.etree.ElementTree as ET
import re


def normalize(word):
    return word.lower().replace("ё", "е")


def load_dictionary(path):
    """
    Загружает dict.opcorpora.xml
    Возвращает словарь: словоформа -> (лемма, POS)
    """
    result = {}

    context = ET.iterparse(path, events=("end",))
    for event, elem in context:
        if elem.tag == "lemma":
            l = elem.find("l")
            if l is None:
                elem.clear()
                continue

            lemma_word = normalize(l.attrib["t"])

            # первая граммема — часть речи
            pos = None
            g = l.find("g")
            if g is not None:
                pos = g.attrib["v"]

            if pos is None:
                elem.clear()
                continue

            # добавляем все словоформы
            for f in elem.findall("f"):
                word = normalize(f.attrib["t"])
                result[word] = (lemma_word, pos)

            elem.clear()

    return result


def levenshtein_limited(a, b, max_dist):
    """
    Вычисляет расстояние Левенштейна.
    Если расстояние > max_dist — прекращает расчёт.
    """
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        curr = [i]
        min_row = i

        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1

            val = min(
                prev[j] + 1,      # удаление
                curr[j - 1] + 1,  # вставка
                prev[j - 1] + cost  # замена
            )
            curr.append(val)

            if val < min_row:
                min_row = val

        if min_row > max_dist:
            return max_dist + 1

        prev = curr

    return prev[-1]


def find_closest(word, dictionary, max_dist=float("+inf")):
    best_word = None
    best_dist = max_dist + 1

    for candidate in dictionary.keys():
        dist = levenshtein_limited(word, candidate, best_dist)
        if dist < best_dist:
            best_dist = dist
            best_word = candidate

            if dist == 0:
                break

    return best_word, best_dist


def word_accuracy(a, b):
    """
    Точность = угаданные буквы / max(len(a), len(b))
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0

    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / max_len


def process_text(text, dictionary):
    lines = text.strip().split("\n")
    output_lines = []

    total_accuracy = 0
    word_count = 0

    for line in lines:
        clean = re.sub(r"[.,!?]", "", line)
        tokens = clean.split()

        parts = []

        for token in tokens:
            word_count += 1
            w = normalize(token)

            if w in dictionary:
                lemma, pos = dictionary[w]
                parts.append(f"{token}{{{lemma}={pos}}}")
                total_accuracy += 1.0
            else:
                closest, dist = find_closest(w, dictionary)

                if closest:
                    lemma, pos = dictionary[closest]
                    parts.append(f"{token}{{{lemma}={pos}}}")
                    total_accuracy += word_accuracy(w, closest)
                else:
                    parts.append(f"{token}{{?=?}}")
                    total_accuracy += 0.0

        output_lines.append(" ".join(parts))

    avg_accuracy = total_accuracy / word_count if word_count > 0 else 0.0
    return "\n".join(output_lines), avg_accuracy


def test_run(test_text, dictionary):
    result, acc = process_text(test_text, dictionary)
    print("Тестовый запуск для:")
    print(test_text)
    print("Результат тестового запуска:")
    print(result)
    print("Accuracy:", acc)
    

if __name__ == "__main__":
    dictionary = load_dictionary("dict.opcorpora.xml")

    test_text_1 = """Стала стабильнее экономическая и политическая обстановка,
    предприятия вывели из тени зарплаты сотрудников."""

    test_text_2 = "Все Гришины одноклассники уже побывали за границей, он был чуть ли не единственным, кого не вывозили никуда дальше Красной Пахры."

    test_text_3 = "рофлить, кринжевать, депнуть, пофиксить."

    test_run(test_text_1, dictionary)
    test_run(test_text_2, dictionary)
    test_run(test_text_3, dictionary)

    while True:
        result, acc = process_text(input("Введите текст или нажмите Ctrl+C: "), dictionary)
        print("Результат: ", result)
        print("Accuracy:", acc)
