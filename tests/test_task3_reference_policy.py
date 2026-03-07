import unittest

from src.task3_reference_policy import normalize_task3_object_id


class TestNormalizeTask3ObjectId(unittest.TestCase):
    def test_bool_and_none(self):
        self.assertIsNone(normalize_task3_object_id(None))
        self.assertIsNone(normalize_task3_object_id(True))
        self.assertIsNone(normalize_task3_object_id(False))

    def test_integers(self):
        # Valid positive integers
        self.assertEqual(normalize_task3_object_id(0), 0)
        self.assertEqual(normalize_task3_object_id(1), 1)
        self.assertEqual(normalize_task3_object_id(42), 42)

        # Invalid negative integers
        self.assertIsNone(normalize_task3_object_id(-1))
        self.assertIsNone(normalize_task3_object_id(-100))

    def test_floats(self):
        # Valid whole-number positive floats
        self.assertEqual(normalize_task3_object_id(0.0), 0)
        self.assertEqual(normalize_task3_object_id(1.0), 1)
        self.assertEqual(normalize_task3_object_id(42.0), 42)

        # Invalid negative whole-number floats
        self.assertIsNone(normalize_task3_object_id(-1.0))

        # Invalid fractional floats
        self.assertIsNone(normalize_task3_object_id(1.5))
        self.assertIsNone(normalize_task3_object_id(3.14))
        self.assertIsNone(normalize_task3_object_id(-2.5))

    def test_strings(self):
        # Valid numeric strings
        self.assertEqual(normalize_task3_object_id("0"), 0)
        self.assertEqual(normalize_task3_object_id("1"), 1)
        self.assertEqual(normalize_task3_object_id(" 42 "), 42)

        # Invalid numeric strings
        self.assertIsNone(normalize_task3_object_id("-1"))
        self.assertIsNone(normalize_task3_object_id("3.14"))

        # Empty or whitespace strings
        self.assertIsNone(normalize_task3_object_id(""))
        self.assertIsNone(normalize_task3_object_id("   "))

        # Non-numeric strings
        self.assertIsNone(normalize_task3_object_id("abc"))
        self.assertIsNone(normalize_task3_object_id("12a"))

    def test_other_objects(self):
        self.assertIsNone(normalize_task3_object_id([]))
        self.assertIsNone(normalize_task3_object_id({}))
        self.assertIsNone(normalize_task3_object_id(object()))

if __name__ == '__main__':
    unittest.main()
