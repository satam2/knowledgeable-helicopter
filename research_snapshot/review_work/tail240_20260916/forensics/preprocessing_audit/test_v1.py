import unittest
from collections import Counter
import run_v1 as audit


class CategoryAuditTests(unittest.TestCase):
    def test_collision_and_missing_separation(self):
        counts=Counter({('str',' A '):2,('str','A'):3,('str','a'):4,('null',''):5,('str','NULL'):6})
        result=audit.summarize(counts)
        self.assertEqual(result['null_rows'],5)
        self.assertEqual(result['normalizations']['trim']['changed_rows'],2)
        self.assertEqual(result['normalizations']['trim']['collision_groups'],1)
        self.assertEqual(result['normalizations']['ascii_case']['collision_rows'],7)
        self.assertEqual(result['flags']['missing_like_literal_unconfirmed'],6)
        self.assertEqual(sum(counts.values()),20)

    def test_numeric_and_unicode_are_diagnostic_only(self):
        counts=Counter({('str','01'):2,('str','1'):3,('str','1.0'):4,('str','\uff11'):5})
        result=audit.summarize(counts)
        self.assertEqual(result['normalizations']['numeric_hypothesis']['collision_rows'],9)
        self.assertEqual(result['normalizations']['nfkc']['collision_rows'],8)
        self.assertEqual(counts[('str','01')],2)
        self.assertEqual(audit.normalize('04L','numeric_hypothesis'),'04L')


if __name__=='__main__':unittest.main()
