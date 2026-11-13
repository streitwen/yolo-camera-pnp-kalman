from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from detect_pnp_kalman import build_parser, scaled_size


class CliTests(unittest.TestCase):
    def test_config_file_sets_pnp_and_kalman_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "assignment.json"
            path.write_text(
                json.dumps(
                    {
                        "armor_width_mm": 111.0,
                        "armor_height_mm": 44.0,
                        "corner_order": "TR,TL,BL,BR",
                        "prediction_seconds": 0.15,
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser(["--config", str(path)]).parse_args(
                ["--config", str(path)]
            )

        self.assertEqual(args.armor_width_mm, 111.0)
        self.assertEqual(args.armor_height_mm, 44.0)
        self.assertEqual(args.corner_order, "TR,TL,BL,BR")
        self.assertEqual(args.prediction_seconds, 0.15)

    def test_scaled_size_keeps_even_dimensions(self):
        self.assertEqual(scaled_size(3840, 2160, 0.5), (1920, 1080))
        self.assertEqual(scaled_size(1441, 1081, 0.5), (720, 540))

    def test_confidence_option_is_not_parsed_as_config_path(self):
        args = build_parser(["--conf", "0.2"]).parse_args(
            ["--conf", "0.2"]
        )
        self.assertEqual(args.conf, 0.2)
        self.assertEqual(args.config, "")


if __name__ == "__main__":
    unittest.main()
