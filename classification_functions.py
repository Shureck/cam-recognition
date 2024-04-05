from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def clip_extract(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, text: list) -> tuple:
    """
    extract features from an image using the CLIP model
    """
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]
    probs_to_text = {text[k]: v for k, v in enumerate(probs)}
    max_value_pair = max(probs_to_text.items(), key=lambda x: x[1])
    return max_value_pair


def process_queries(model:CLIPModel, processor: CLIPProcessor, image: Image.Image, queries: dict) -> dict:
    results = {}
    for key, query in queries.items():
        result = clip_extract(model, processor, image, query)
        results[key] = result[0]
    return results


def extract_features(image: Image.Image) -> dict:
    """extract face features from an image
    returns a dictionary of features: features with 'have' is boolean (1/0), other features are strings
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    queries = {
        "bald_hair": ["bald", "with hair"],
        "skin_color": ["white skin", "fair skin", "dark skin"],
        "eye_color": ["brown eye color", "blue eye color", "green eye color", "grey eye color"],
        "glasses": ["have glasses", "no glasses"],
        "beard": ["have beard", "no beard"],
        "mustache": ["have mustache", "no mustache"]
    }
    results = process_queries(model, processor, image, queries)
    bald_hair = results["bald_hair"]
    have_hair = bald_hair != "bald"
    skin_color = results["skin_color"]
    eye_color = results["eye_color"]
    glasses = results["glasses"]
    have_glasses = glasses != "no glasses"
    beard = results["beard"]
    have_beard = beard != "no beard"
    mustache = results["mustache"]
    have_mustache = mustache != "no mustache"
    if have_hair:
        queries = {
            "hair_type": ["curly hair", "straight hair"],
            "hair_length": ["short hair", "long hair"],
            "hair_color": ["black hair", "bright hair", "brown hair", "blond hair", "red hair", "blue hair", "green hair", "purple hair"],
        }
        results = process_queries(model, processor, image, queries)
        hair_color = results["hair_color"]
        hair_type = results["hair_type"]
        hair_length = results["hair_length"]
    else:
        hair_color = "None"
        hair_type = "None"
        hair_length = "None"
    return {
        "have_hair": int(have_hair), "hair_color": hair_color, "hair_type": hair_type, "hair_length": hair_length,
        "eye_color": eye_color, "have_glasses": int(have_glasses), "have_beard": int(have_beard),
        "have_mustache": int(have_mustache), "skin_color": skin_color
    }


def test_extract_features():
    import os
    path = os.path.join("photo", "hair", "test", "other")
    file_list = os.listdir(path)
    for file in file_list:
        print(file)
        image = Image.open(os.path.join(path, file))
        print(extract_features(image))


if __name__ == "__main__":
    print("Testing features extraction from images")
    image = Image.open("ttest.jpg")
    print(extract_features(image))