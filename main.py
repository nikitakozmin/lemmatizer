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


def process_text(text, dictionary):
    lines = text.strip().split("\n")
    output_lines = []

    total = 0
    found = 0

    for line in lines:
        # убираем только допустимые знаки препинания
        clean = re.sub(r"[.,!?]", "", line)
        tokens = clean.split()

        parts = []
        for token in tokens:
            total += 1
            w = normalize(token)

            if w in dictionary:
                lemma, pos = dictionary[w]
                parts.append(f"{token}{{{lemma}={pos}}}")
                found += 1
            else:
                parts.append(f"{token}{{?=?}}")

        output_lines.append(" ".join(parts))

    accuracy = found / total if total > 0 else 0.0
    return "\n".join(output_lines), accuracy


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
