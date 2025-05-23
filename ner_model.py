from transformers import pipeline

# Load the NER pipeline
nlp_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

def extract_entities(text):
    entities = nlp_pipeline(text)
    results = {"People": [], "Organizations": [], "Locations": []}

    for entity in entities:
        word = entity["word"].strip()
        entity_type = entity["entity_group"]

        if entity_type == "PER":
            results["People"].append(word)
        elif entity_type == "ORG":
            results["Organizations"].append(word)
        elif entity_type == "LOC":
            results["Locations"].append(word)

    return results
