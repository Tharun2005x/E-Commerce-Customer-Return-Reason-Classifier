import os
import random
import pandas as pd
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

TARGET_CLASSES = ["Sizing", "Damaged Item", "Wrong Product", "Late Delivery"]

KEYWORD_RULES = {
    "Sizing": ["size", "small", "large", "tight", "loose", "fit", "short", "long", "petite", "snug", "baggy"],
    "Damaged Item": ["torn", "rip", "hole", "broken", "stain", "defect", "cheap", "fell apart", "zipper", "damaged"],
    "Wrong Product": ["wrong", "different color", "not what i ordered", "mistake", "incorrect", "sent me"],
    "Late Delivery": ["late", "shipping", "delayed", "arrived after", "slow delivery", "took weeks", "never arrived"],
}

SYNTHETIC_TEMPLATES = {
    "Sizing": [
        "Ordered my usual medium but it runs way too small around the chest and shoulders.",
        "Way too baggy and oversized even after sizing down. Returning for a smaller fit.",
        "The sleeves are too short and the waist is uncomfortably tight.",
        "True to size in the hips but massive in the bust, doesn't fit right at all.",
        "I am 5'4 and this dress drags on the floor, way too long for petite frames.",
    ],
    "Damaged Item": [
        "Arrived with a huge rip along the side seam and loose threads everywhere.",
        "The zipper broke the very first time I tried it on. Defective item.",
        "There is a noticeable ink stain on the front pocket right out of the package.",
        "Fabric arrived torn near the collar and buttons were missing.",
        "Package was crushed and the product inside was scratched and damaged.",
    ],
    "Wrong Product": [
        "I ordered the navy blue blazer in size M, but received a red cardigan instead.",
        "Completely wrong item sent in the parcel, not even close to what I ordered.",
        "Received size XL instead of XS as listed on my invoice.",
        "Wrong color delivered—ordered emerald green and got plain black.",
        "The package had someone else's order inside, totally wrong product.",
    ],
    "Late Delivery": [
        "Ordered this for a wedding last weekend and it arrived 6 days past the promised date.",
        "Shipping was severely delayed for over three weeks so I bought another outfit locally.",
        "Package sat in transit forever and arrived way too late for my trip.",
        "Express shipping failed and the item showed up two weeks late.",
        "Delivered well after the event date despite paying for priority delivery.",
    ],
}

def classify_review(text: str):
    t = text.lower()
    for label, kws in KEYWORD_RULES.items():
        if any(kw in t for kw in kws):
            return label
    return None

def build_dataset():
    rows = []
    try:
        ds = load_dataset("Censius-AI/ECommerce-Women-Clothing-Reviews", split="train")
        for item in ds:
            text = str(item.get("Review Text") or "").strip()
            rating = item.get("Rating", 5)
            if len(text) < 20 or rating > 3:
                continue
            label = classify_review(text)
            if label:
                rows.append({"comment": text, "return_reason": label, "source": "hf_reviews"})
            if len(rows) >= 1200:
                break
    except Exception as e:
        print(f"Dataset fetch fallback triggered: {e}")

    df_real = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["comment", "return_reason", "source"])
    balanced_rows = []
    target_per_class = 300

    for label in TARGET_CLASSES:
        subset = df_real[df_real["return_reason"] == label].head(target_per_class).to_dict("records")
        while len(subset) < target_per_class:
            base = random.choice(SYNTHETIC_TEMPLATES[label])
            suffix = random.choice([
                " Please process my refund.",
                " Need to return immediately.",
                " Very disappointed with this order.",
                " Initiating return request today.",
                "",
            ])
            subset.append({"comment": base + suffix, "return_reason": label, "source": "synthetic_return"})
        balanced_rows.extend(subset)

    df_final = pd.DataFrame(balanced_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv("data/return_reasons.csv", index=False)
    print(f"Saved {len(df_final)} balanced rows to data/return_reasons.csv")
    print(df_final["return_reason"].value_counts())

if __name__ == "__main__":
    build_dataset()
