from src import api
from PIL import Image
import io
import asyncio


def test_home_and_version():
    r = api.home()
    assert isinstance(r, dict)
    assert "message" in r

    v = api.version()
    assert isinstance(v, dict)
    assert v.get("framework") == "FastAPI"


def test_health():
    h = api.health()
    assert isinstance(h, dict)
    assert "status" in h
    assert "model_loaded" in h


def test_predict_endpoint_behaviour():
    # create a tiny RGB image in-memory
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    class DummyFile:
        def __init__(self, fileobj, filename="test.jpg"):
            self.file = fileobj
            self.filename = filename

    dummy = DummyFile(buf)

    # call the async predict handler directly
    try:
        r = asyncio.run(api.predict(dummy))
    except Exception as e:
        # If model not loaded, the API raises HTTPException (mapped to 500 in server).
        # For our direct call, ensure an exception is raised when model is missing.
        assert not api.model_loaded
        return

    if api.model_loaded:
        assert isinstance(r, dict)
        assert "filename" in r
        assert "inference_time_ms" in r
    else:
        # should not reach here
        assert False, "predict did not raise when model not loaded"
