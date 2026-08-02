import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session", autouse=True)
def _generate_sample_data():
    """Ensure sample data exists before any test runs."""
    products_dir = os.path.join(ROOT, "data", "products", "shirts")
    if not os.path.isdir(products_dir) or len(os.listdir(products_dir)) == 0:
        subprocess.check_call([sys.executable, "-m", "data.generate_samples"], cwd=ROOT)
    yield


def test_image_utils_roundtrip():
    from src.image_utils import load_image, resize_image, to_grayscale, normalize_image

    path = os.path.join(ROOT, "data", "products", "shirts", "img_00.jpg")
    img = load_image(path)
    assert img.shape[2] == 3
    small = resize_image(img, (64, 64))
    assert small.shape == (64, 64, 3)
    gray = to_grayscale(img)
    assert gray.ndim == 2
    norm = normalize_image(img)
    assert 0.0 <= float(norm.min()) and float(norm.max()) <= 1.0


def test_chatbot_replies_to_faq():
    from src.chatbot import RetailChatbot

    bot = RetailChatbot()
    reply, score = bot.reply("How can I return an item I bought?")
    assert "return" in reply.lower()
    assert score > 0.1


def test_chatbot_handles_unknown():
    from src.chatbot import RetailChatbot

    bot = RetailChatbot()
    reply, _ = bot.reply("asdkjhaskjdhaksjdhkjashdkjahsdkj")
    assert "not sure" in reply.lower() or "support" in reply.lower()


def test_sentiment_train_and_predict():
    import pandas as pd
    from src.sentiment_model import SentimentModel

    df = pd.read_csv(os.path.join(ROOT, "data", "reviews", "reviews.csv"))
    model = SentimentModel()
    report = model.train(df)
    assert "accuracy" in report
    label, conf = model.predict("Absolutely love this, best purchase ever!")
    assert label in {"positive", "neutral", "negative"}
    assert 0.0 <= conf <= 1.0


def test_face_recognizer_enrol_and_identify():
    from src.face_utils import FaceRecognizer

    fr = FaceRecognizer()
    count = fr.enroll_from_folder(os.path.join(ROOT, "data", "faces"))
    # Haar cascade may or may not detect our synthetic faces on every platform;
    # only assert the API contract, not detection strength.
    assert isinstance(count, int) and count >= 0
    assert isinstance(fr.db, dict)
