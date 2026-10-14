import subprocess
import unittest
from pathlib import Path

import installer_config


class InstallerConfigTests(unittest.TestCase):
    def test_codex_merge_replaces_only_managed_block(self):
        entry = installer_config.build_server_entry("/bin/python", "/tmp/server.py", "/tmp/home")
        first = installer_config.merge_codex_config("[profiles.dev]\nmodel = \"x\"\n", "rag", entry)
        second = installer_config.merge_codex_config(first, "rag", entry)

        self.assertIn('[mcp_servers.rag]', second)
        self.assertIn('[profiles.dev]', second)
        self.assertEqual(second.count("BEGIN confluence-rag-mcp"), 1)

    def test_codex_merge_removes_legacy_server_and_nested_tables(self):
        entry = installer_config.build_server_entry("/bin/python", "/tmp/server.py", "/tmp/home")
        existing = """[mcp_servers.go-confluence-kb]
command = \"/bin/python\"

[mcp_servers.go-confluence-kb.env]
HF_HOME = \"/tmp/old-models\"

[profiles.dev]
model = \"x\"
"""

        merged = installer_config.merge_codex_config(existing, "confluence-rag", entry)

        self.assertNotIn("go-confluence-kb", merged)
        self.assertIn('[profiles.dev]', merged)
        self.assertIn('[mcp_servers.confluence-rag]', merged)

    def test_help_names_codex_and_its_two_config_files(self):
        script = Path(__file__).resolve().parents[1] / "install.sh"

        result = subprocess.run(
            ["bash", str(script), "--help"], check=False, capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Codex", result.stdout)
        self.assertIn("~/.o3-cli/o3_plugin_agent/codex/config.toml", result.stdout)


if __name__ == "__main__":
    unittest.main()
