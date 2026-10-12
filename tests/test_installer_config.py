import unittest

import installer_config


class InstallerConfigTests(unittest.TestCase):
    def test_codex_merge_replaces_only_managed_block(self):
        entry = installer_config.build_server_entry("/bin/python", "/tmp/server.py", "/tmp/home")
        first = installer_config.merge_codex_config("[profiles.dev]\nmodel = \"x\"\n", "rag", entry)
        second = installer_config.merge_codex_config(first, "rag", entry)

        self.assertIn('[mcp_servers.rag]', second)
        self.assertIn('[profiles.dev]', second)
        self.assertEqual(second.count("BEGIN confluence-rag-mcp"), 1)


if __name__ == "__main__":
    unittest.main()
