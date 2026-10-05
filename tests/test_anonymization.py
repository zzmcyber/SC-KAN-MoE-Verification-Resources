import unittest
from datetime import datetime,timedelta
from preprocessing.anonymization import station_anchors,site_anchor_mapping,shifted_time,parse_time,iso_time


class AnonymizationTests(unittest.TestCase):
    def test_reproducible_distinct_anchors(self):
        anchors=station_anchors(12)
        self.assertEqual(anchors,station_anchors(12))
        self.assertEqual(len(set(anchors)),12)
        self.assertTrue(all(datetime(2001,1,1)<=x<datetime(2011,1,1) for x in anchors))

    def test_exact_intervals_and_timezone(self):
        first=datetime(2021,4,5,9,30,0,123000);anchor=station_anchors(1)[0]
        for delta in [timedelta(minutes=-20),timedelta(days=-6),timedelta(days=390,seconds=1)]:
            self.assertEqual(shifted_time(first+delta,first,anchor)-anchor,delta)
        self.assertEqual(parse_time('2021-04-05T17:30:00.123000+08:00'),first)
        self.assertEqual(parse_time(iso_time(first)),first)

    def test_anchor_mapping_ignores_input_order(self):
        sites=[f'SITE_{i:03d}' for i in range(1,13)]
        expected=site_anchor_mapping(sites)
        self.assertEqual(expected,site_anchor_mapping(reversed(sites)))
        self.assertEqual(expected,site_anchor_mapping(set(sites)))
        self.assertEqual(expected,site_anchor_mapping(sites[5:]+sites[:5]))
        for site in sites:
            self.assertEqual(site_anchor_mapping([site])[site],expected[site])

    def test_anchor_mapping_rejects_invalid_ids(self):
        for sites in [['SITE_013'],['SITE_001','SITE_001'],['private-site']]:
            with self.assertRaises(ValueError):site_anchor_mapping(sites)


if __name__=='__main__':unittest.main()
