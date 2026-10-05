import csv
import json
from pathlib import Path
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_scope_is_explicit(self):
        manifest=json.loads((ROOT/'metadata/release_manifest.json').read_text())
        self.assertEqual(manifest['release_scope'],'anonymized_auxiliary_inputs_and_code')
        self.assertEqual(manifest['anonymous_sites'],12)
        self.assertGreater(manifest['auxiliary_records'],0)
        self.assertTrue((ROOT/'data/auxiliary_subset').is_dir())

    def test_parameter_consistency(self):
        reported=yaml.safe_load((ROOT/'configs/documented_settings.yaml').read_text())['reported_settings']
        architecture=yaml.safe_load((ROOT/'configs/reference_architecture_settings.yaml').read_text())
        ranges=yaml.safe_load((ROOT/'configs/hyperparameter_search_space.yaml').read_text())['candidate_ranges']
        self.assertEqual(reported,{'num_experts':6,'top_k':3})
        self.assertEqual(architecture['reported_settings'],reported)
        self.assertEqual(ranges['top_k'],[2,3])
        self.assertGreaterEqual(ranges['num_experts']['minimum'],max(ranges['top_k']))
        self.assertEqual(architecture['reference_settings']['spline_order'],3)

    def test_dictionary_unique_and_explicit(self):
        with (ROOT/'metadata/feature_dictionary.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        keys=[(r['table'],r['field']) for r in rows]
        self.assertEqual(len(keys),len(set(keys)))
        allowed={'processed_input','derived','anonymized_timestamp','anonymous_identifier'}
        for row in rows:
            self.assertIn(row['processing_state'],allowed)
            self.assertTrue(row['description']);self.assertTrue(row['source'])
            self.assertTrue(row['unit'])

    def test_no_weights_or_private_artifacts_in_distribution(self):
        ignored_roots={'.git','local_only','__pycache__','.venv'}
        forbidden_suffixes={'.pt','.pth','.ckpt','.xlsx','.xls','.pkl','.pem','.p12','.pfx'}
        for path in ROOT.rglob('*'):
            if any(part in ignored_roots for part in path.relative_to(ROOT).parts):continue
            if path.is_file():self.assertNotIn(path.suffix.lower(),forbidden_suffixes,str(path))
        ignore=(ROOT/'.gitignore').read_text()
        self.assertIn('local_only/',ignore)
        self.assertIn('station_mapping*',ignore)


if __name__=='__main__':unittest.main()
