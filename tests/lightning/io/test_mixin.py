from nemo.lightning import io


class DummyClass(io.IOMixin):
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


class TestIOMixin:
    def test_reinit(self):
        dummy = DummyClass(5, 5)
        copied = io.reinit(dummy)
        assert copied is not dummy
        assert copied.a == dummy.a
        assert copied.b == dummy.b
