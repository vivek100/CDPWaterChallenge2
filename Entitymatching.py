
from __future__ import unicode_literals, print_function

import plac
import spacy


TEXTS = [
    "BizLink has 9 facilities located in China. 3 facilities in Kunshan and Shenzen are disclosed. 6 facilities are excluded.",
    "Sales offices and research lab in Springfield, MO"
]


@plac.annotations(
    model=("Model to load (needs parser and NER)", "positional", None, str)
)
def main(model="en_core_web_sm"):
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
    print("Processing %d texts" % len(TEXTS))

    for text in TEXTS:
        doc = nlp(text)
        relations = extract_location_relations(doc)
        for r1, r2 in relations:
            print("{:<10}\t{}\t{}".format(r1.text, r2.ent_type_, r2.text))


def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def extract_location_relations(doc):
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    relations = []
    for location in filter(lambda w: w.ent_type_ == "GPE", doc):
        if location.dep_ in ("attr", "dobj"):
            subject = [w for w in location.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, location))
        elif location.dep_ == "pobj" and location.head.dep_ == "prep":
            relations.append((location.head.head, location))
    return relations


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Net income      location   $9.4 million
    # the prior year  location   $2.7 million
    # Revenue         location   twelve billion dollars
    # a loss          location   1b
