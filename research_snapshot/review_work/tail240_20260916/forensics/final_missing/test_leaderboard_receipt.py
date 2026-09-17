import unittest
import leaderboard_receipt as subject


class LeaderboardTests(unittest.TestCase):
    def row(self,identifier,team,filename,score):
        return dict(submissionId=identifier,teamName=team,filename=filename,score=score,usedPairs=344841)

    def test_requires_v3_before_reporting_rank(self):
        rows=[self.row('old',subject.TEAM,'knowledgeable-helicopter_v2.parquet',294.626)]
        with self.assertRaisesRegex(ValueError,'V3'):
            subject.summarize(rows)

    def test_best_per_team_rank_and_ties(self):
        rows=[self.row('v3',subject.TEAM,subject.KEY,280.),
              self.row('old',subject.TEAM,'old.parquet',294.),
              self.row('other1','other','a.parquet',270.),
              self.row('other2','other','b.parquet',290.),
              self.row('equal','tied','c.parquet',280.)]
        result=subject.summarize(rows)
        self.assertEqual(result['team_rank_by_best_score'],2)
        self.assertEqual(result['teams'],3)
        self.assertEqual(result['v3']['submissionId'],'v3')

    def test_duplicate_page_boundary_and_unaccepted_v3_fail(self):
        row=self.row('v3',subject.TEAM,subject.KEY,280.)
        with self.assertRaisesRegex(ValueError,'page'):
            subject.summarize([row,row])
        row['score']=None
        with self.assertRaisesRegex(ValueError,'V3'):
            subject.summarize([row])


if __name__=='__main__':
    unittest.main()
