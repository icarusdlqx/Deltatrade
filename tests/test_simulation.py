import os
import unittest
from pathlib import Path

from v2.orchestrator import run_once
from v2.settings_bridge import get_cfg


class SimulationRunTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = {k: os.environ.get(k) for k in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "SIM_MODE"]}
        os.environ.pop("ALPACA_API_KEY", None)
        os.environ.pop("ALPACA_SECRET_KEY", None)
        os.environ["SIM_MODE"] = "true"

        cfg = get_cfg()
        self.episodes_path = Path(cfg.EPISODES_PATH)
        self.state_path = Path("data/sim_trading_state.json")
        self._episodes_backup = self.episodes_path.read_text(encoding="utf-8") if self.episodes_path.exists() else None
        self._state_backup = self.state_path.read_text(encoding="utf-8") if self.state_path.exists() else None
        if self.episodes_path.exists():
            self.episodes_path.unlink()
        if self.state_path.exists():
            self.state_path.unlink()

    def tearDown(self):
        for key, val in self._env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        if self._episodes_backup is not None:
            self.episodes_path.write_text(self._episodes_backup, encoding="utf-8")
        elif self.episodes_path.exists():
            self.episodes_path.unlink()
        if self._state_backup is not None:
            self.state_path.write_text(self._state_backup, encoding="utf-8")
        elif self.state_path.exists():
            self.state_path.unlink()

    def test_run_once_generates_episode_in_simulation(self):
        result = run_once()
        self.assertTrue(result.get("simulated"), "Expected simulated flag when running without credentials")
        self.assertIn("orders_submitted", result)
        self.assertTrue(self.episodes_path.exists(), "Episodes log should be created")
        text = self.episodes_path.read_text(encoding="utf-8")
        self.assertGreater(len(text.strip()), 0, "Episodes log should contain data")


if __name__ == "__main__":
    unittest.main()
