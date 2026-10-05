import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import re
import unittest
from preprocessing.anonymization import parse_time,site_anchor_mapping

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/auxiliary_subset'
TABLES=['sar_features','meteorological_features','precipitation_features',
        'vegetation_features','terrain_features','temporal_matching']


def read(path):
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)):
            raise ValueError('Expected unique nonempty CSV headers')
        rows=list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError('Expected nonempty rectangular CSV records')
    return rows


class AuxiliaryDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Required public files: absence is an error, never a skipped test.
        cls.tables={name:read(BASE/(name+'.csv')) for name in TABLES}
        cls.combined=read(BASE/'combined/sc_kan_moe_auxiliary_features.csv')
        cls.summary=read(BASE/'site_summary.csv')
        cls.manifest=json.loads((ROOT/'metadata/release_manifest.json').read_text())
        cls.dictionary=read(ROOT/'metadata/feature_dictionary.csv')
        cls.all_tables={**{k+'.csv':v for k,v in cls.tables.items()},
                        'site_summary.csv':cls.summary,
                        'combined/sc_kan_moe_auxiliary_features.csv':cls.combined}

    def test_sites_and_unique_sample_ids(self):
        expected={f'SITE_{i:03d}' for i in range(1,13)}
        self.assertEqual({r['site_id'] for r in self.combined},expected)
        self.assertEqual({r['site_id'] for r in self.summary},expected)
        self.assertEqual(len(self.summary),12)
        for name,rows in {**self.tables,'combined':self.combined}.items():
            ids=[row['sample_id'] for row in rows]
            self.assertEqual(len(ids),len(set(ids)),name)
            by_site=defaultdict(list)
            for row in rows:
                self.assertIn(row['site_id'],expected)
                self.assertRegex(row['sample_id'],r'^SITE_\d{3}_S\d{4,}$')
                self.assertTrue(row['sample_id'].startswith(row['site_id']+'_'))
                by_site[row['site_id']].append(row['sample_id'])
            for site,values in by_site.items():
                self.assertEqual(values,[f'{site}_S{i:04d}' for i in range(1,len(values)+1)])

    def test_exact_one_to_one_join(self):
        expected={r['sample_id'] for r in self.combined}
        indices={}
        for name,rows in self.tables.items():
            self.assertEqual(len(rows),len(self.combined),name)
            self.assertEqual({r['sample_id'] for r in rows},expected,name)
            indices[name]={r['sample_id']:r for r in rows}
        for row in self.combined:
            merged={}
            for index in indices.values():
                for field,value in index[row['sample_id']].items():
                    if field in merged:self.assertEqual(merged[field],value,field)
                    merged[field]=value
            self.assertEqual(merged,row)

    def test_numeric_values_are_finite(self):
        for definition in self.dictionary:
            unit=definition['unit'];field=definition['field']
            if unit in ('identifier','ISO 8601','category'):continue
            for row in self.all_tables[definition['table']]:
                value=row[field]
                if value in ('True','False'):number=float(value=='True')
                else:number=float(value)
                self.assertTrue(math.isfinite(number),field)
                if field in ('s1_found','s1_core_valid','s1_lia_valid','s1_dem_valid','freeze_thaw_flag'):
                    self.assertIn(number,(0.,1.))
        for row in self.tables['sar_features']:
            self.assertEqual(row['s1_quality_flag'],'A_core_lia_dem_valid')

    def test_public_field_allowlist(self):
        # These schema fields contain only input values, QC labels and translated times.
        forbidden=re.compile(r'observed|predicted|^sm$|sm_lag|delta_sm|antecedent|observation|ismn|latitude|longitude|^lat$|^lon$|country|region|network|climate|land.?cover|row.?id|orbit|platform|image.?id|^date$|^timestamp$|source.*row|private',re.I)
        for name,rows in self.all_tables.items():
            for field in rows[0]:
                self.assertIsNone(forbidden.search(field),f'{name}:{field}')
                self.assertNotEqual(field,'SITE_ID')
        for path in (ROOT/'data').rglob('*.csv'):
            self.assertIn(path.relative_to(BASE).as_posix(),self.all_tables)
        self.assertFalse((ROOT/'data/released_subset').exists())
        self.assertFalse(list(BASE.rglob('observations.csv')))
        self.assertFalse(list(BASE.rglob('predictions.csv')))

    def test_temporal_order_intervals_and_stable_anchors(self):
        groups=defaultdict(list)
        for row in self.tables['temporal_matching']:
            reference=parse_time(row['anonymous_reference_time'])
            sar=parse_time(row['anonymous_sar_time'])
            self.assertAlmostEqual((sar-reference).total_seconds()/60,float(row['s1_time_diff_minutes']),places=6)
            groups[row['site_id']].append(reference)
        anchors=site_anchor_mapping(reversed(sorted(groups)))
        self.assertEqual(anchors,site_anchor_mapping(set(groups)))
        for site,times in groups.items():
            self.assertEqual(times,sorted(times))
            self.assertEqual(len(times),len(set(times)))
            self.assertEqual(times[0],anchors[site])
        for row in self.summary:
            times=groups[row['site_id']]
            self.assertEqual(int(row['auxiliary_record_count']),len(times))
            self.assertEqual(parse_time(row['first_anonymous_time']),times[0])
            self.assertEqual(parse_time(row['last_anonymous_time']),times[-1])

    def test_precipitation_window_order(self):
        for row in self.tables['precipitation_features']:
            windows=[float(row['precip_'+w]) for w in ['1h','3h','6h','24h','3d','7d']]
            self.assertTrue(all(v>=0 for v in windows))
            self.assertTrue(all(a<=b+1e-8 for a,b in zip(windows,windows[1:])))
            self.assertGreaterEqual(float(row['hours_since_rain']),0.)

    def test_vegetation_ranges(self):
        for row in self.tables['vegetation_features']:
            for field in ('NDVI','EVI'):
                self.assertGreaterEqual(float(row[field]),-1.)
                self.assertLessEqual(float(row[field]),1.)
            self.assertGreaterEqual(float(row['LAI']),0.)
            self.assertGreaterEqual(float(row['FPAR']),0.)
            self.assertLessEqual(float(row['FPAR']),1.)

    def test_sar_ranges_and_matching_interval(self):
        for row in self.tables['sar_features']:
            self.assertGreaterEqual(float(row['IncAngle']),0.)
            self.assertLessEqual(float(row['IncAngle']),90.)
            for field in ('VV_single','VH_single','VV_9pix','VH_9pix'):
                self.assertGreater(float(row[field]),0.)
        # A bound for the published reference/acquisition pairs, not an extraction rule.
        for row in self.tables['temporal_matching']:
            self.assertLessEqual(abs(float(row['s1_time_diff_minutes'])),1.)

    def test_nonnegative_wind_speed(self):
        for row in self.tables['meteorological_features']:
            self.assertGreaterEqual(float(row['wind_speed_10m']),0.)

    def test_dictionary_matches_csv_exactly(self):
        actual={(name,field) for name,rows in self.all_tables.items() for field in rows[0]}
        declared=[(r['table'],r['field']) for r in self.dictionary]
        self.assertEqual(len(declared),len(set(declared)))
        self.assertEqual(set(declared),actual)
        for row in self.dictionary:
            self.assertTrue(row['unit'].strip(),row['field'])
            self.assertIn(row['processing_state'],{'processed_input','derived','anonymized_timestamp','anonymous_identifier'})

    def test_manifest_counts_and_hashes(self):
        self.assertEqual(self.manifest['release_scope'],'anonymized_auxiliary_inputs_and_code')
        self.assertEqual(self.manifest['anonymous_sites'],12)
        self.assertEqual(self.manifest['auxiliary_records'],len(self.combined))
        for flag in ['in_situ_observations_included','record_level_predictions_included','trained_weights_included',
                     'coordinates_included','location_categories_included','absolute_dates_included','scientific_values_modified']:
            self.assertIs(self.manifest[flag],False)
        self.assertIs(self.manifest['privacy_shifted_timestamps'],True)
        self.assertIs(self.manifest['within_site_intervals_preserved'],True)
        self.assertEqual(set(self.manifest['table_dimensions']),set(self.all_tables))
        for name,rows in self.all_tables.items():
            self.assertEqual(self.manifest['table_dimensions'][name],{'rows':len(rows),'fields':len(rows[0])})
        paths={p.relative_to(ROOT).as_posix():p for p in BASE.rglob('*.csv')}
        self.assertEqual(set(self.manifest['file_sha256']),set(paths))
        for relative,path in paths.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),self.manifest['file_sha256'][relative])


if __name__=='__main__':unittest.main()
