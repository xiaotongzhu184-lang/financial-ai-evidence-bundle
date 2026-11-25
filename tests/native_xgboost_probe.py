"""Small probe used by the project test launcher on native-runtime platforms."""
import unittest


class NativeXgboostProbe(unittest.TestCase):
    def test_xgboost_native_library_imports(self):
        import xgboost
        self.assertTrue(xgboost.__version__)


if __name__ == '__main__':
    unittest.main()
