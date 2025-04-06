from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Text
from rasa.shared.core.events import Event
from rasa.shared.nlu.training_data.message import Message
import torch

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER],
    is_trainable=False
)
class CustomIntentClassifier(GraphComponent):
    @classmethod
    def create(
        cls,
        config: dict,
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "CustomIntentClassifier":
        return cls(config)

    def __init__(self, config: dict):
        model_path = "./intent_classifier"  # Adjust path if needed
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.label_map = {
            0: "atis_abbreviation",
            1: "atis_aircraft",
            2: "atis_airfare",
            3: "atis_airline",
            4: "atis_flight",
            5: "atis_flight_time",
            6: "atis_ground_service",
            7: "atis_quantity",
        }

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            text = message.get("text")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            intent = self.label_map[predicted_class]
            message.set("intent", {"name": intent, "confidence": 0.95}, add_to_output=True)
        return messages

    def train(self, training_data, **kwargs) -> None:
        # No training needed since your model is pre-trained
        pass