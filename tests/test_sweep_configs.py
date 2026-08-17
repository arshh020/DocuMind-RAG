import unittest

from docsrag.config import RerankConfig, Settings
from docsrag.evaluation.retrieval import default_sweep_configs


class TestDefaultSweepConfigs(unittest.TestCase):
    def test_baseline_rows_never_rerank(self) -> None:
        base = Settings()
        base = base.with_overrides(
            rerank=RerankConfig(**{**base.rerank.__dict__, "enabled": True})
        )
        configs = default_sweep_configs(base)
        names = [name for name, _ in configs]
        self.assertIn("hybrid + rerank", names)
        for name, settings in configs:
            if name == "hybrid + rerank":
                self.assertTrue(settings.rerank.enabled, name)
            else:
                self.assertFalse(settings.rerank.enabled, name)

    def test_no_rerank_row_when_disabled(self) -> None:
        configs = default_sweep_configs(Settings())
        self.assertEqual(len(configs), 4)
        self.assertTrue(all(not s.rerank.enabled for _, s in configs))


if __name__ == "__main__":
    unittest.main()
