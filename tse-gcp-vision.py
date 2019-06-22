#!/usr/bin/python3

import io, sys
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
from enum import Enum

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

def detect_document(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))
    return response

def draw_boxes(image, bounds, color,width=5):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        draw.line([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y,
            bound.vertices[0].x, bound.vertices[0].y],fill=color, width=width)
    return image

def get_document_bounds(response, feature, min_confidence=0):
    bounds = []
    document = response.full_text_annotation

    for i,page in enumerate(document.pages):
        for block in page.blocks:
            if feature==FeatureType.BLOCK:
                bounds.append(block.bounding_box)
            for paragraph in block.paragraphs:
                if feature==FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL and symbol.confidence >= min_confidence):
                            bounds.append(symbol.bounding_box)
                    if (feature == FeatureType.WORD and word.confidence >= min_confidence):
                        bounds.append(word.bounding_box)
    return bounds

path = sys.argv[1]

original = Image.open(path)
cropped = original.crop((100, 765, 575, 1935))

cropped_path = path.replace(".jpg", "-cropped.jpg")
cropped.save(cropped_path)

response = detect_document(cropped_path)
bounds=get_document_bounds(response, FeatureType.SYMBOL, .20)
image_with_boxes = draw_boxes(Image.open(cropped_path), bounds, 'yellow')
image_with_boxes.save(cropped_path.replace(".jpg", "-boxes.jpg"))

