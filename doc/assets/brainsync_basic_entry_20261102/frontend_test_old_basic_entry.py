import unittest

import test_decoding


class OldBasicEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = test_decoding.QtWidgets.QApplication.instance() or test_decoding.QtWidgets.QApplication([])

    def setUp(self):
        test_decoding.DecodingWorkflowTests.setUp(self)

    def tearDown(self):
        test_decoding.DecodingWorkflowTests.tearDown(self)

    def test_source_entry_accepts_one_session_and_emits_fixed_basic_profile(self):
        source = self.root / 'source.pt'
        source.write_bytes(b'source identity is checked by the backend')
        self.panel.source_path.setText(str(source))
        self.panel._select_available()
        self.panel.method.setCurrentIndex(self.panel.method.findData('source'))
        self.panel.sync_buttons()
        self.assertTrue(self.panel.calibrate.isEnabled())
        self.assertFalse(self.panel.ocular.isEnabled())
        self.assertFalse(self.panel.epoch_qc.isEnabled())
        self.assertFalse(self.panel.continuous_qc_browse.isEnabled())
        self.assertFalse(self.panel.epoch_qc_browse.isEnabled())
        self.panel.start_calibration()
        arguments = self.runner.calls[-1][1]
        self.assertEqual(arguments[arguments.index('--calibration-profile') + 1], 'old_basic_nn_source')
        self.assertNotIn('--ocular-correction', arguments)
        self.assertNotIn('--continuous-qc-model', arguments)
        self.assertNotIn('--reference-mode', arguments)

    def test_fixed_program_receipt_displays_without_internal_recipe_selection(self):
        self.panel.calibration_results.set_summary({
            'subject_id': 'P001', 'model_kind': 'source_initialized_n2p3',
            'calibration_trials': 45, 'actual_training_trials': 45,
            'internal_validation': {'selected': 'nn_source', 'selection_evidence': False,
                                    'selected_epochs': 90, 'scoring_protocol': 'fixed_full_calibration'},
        })
        self.assertEqual(self.panel.calibration_results.metrics['epochs'].text(), '90')
        self.assertEqual(self.panel.calibration_results.metrics['auc'].text(), '未验证')


if __name__ == '__main__':
    unittest.main()
