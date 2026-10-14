import unittest

import hybrid_knowledge_mcp as rag


class ConfluenceChunkingTests(unittest.TestCase):
    def test_adds_page_title_and_ancestor_breadcrumb_to_each_chunk(self):
        chunks = rag.split_confluence_markdown(
            "# Расчёт\n\nGMV считается ежедневно.\n\n## Исключения\n\nВозвраты исключаются.",
            title="Метрики GMV",
            ancestors=["Команда", "Справочники"],
        )

        self.assertGreaterEqual(len(chunks), 2)
        for text, metadata in chunks:
            self.assertIn("Метрики GMV", text)
            self.assertEqual(metadata["title"], "Метрики GMV")
            self.assertEqual(metadata["ancestor_path"], "Команда > Справочники")

    def test_returns_no_chunks_for_empty_page(self):
        self.assertEqual(
            rag.split_confluence_markdown("   ", title="Пустая", ancestors=[]),
            [],
        )


class PageIndexLifecycleTests(unittest.TestCase):
    def test_page_report_exposes_page_update_status(self):
        report = rag.PageIndexReport(status="indexed")

        self.assertEqual(report.status, "indexed")


class McpApiTests(unittest.TestCase):
    def test_server_uses_confluence_rag_name(self):
        self.assertEqual(rag.SERVER_NAME, "confluence-rag")

    def test_exposes_confluence_only_tools(self):
        names = {item["name"] for item in rag.TOOL_DEFINITIONS}

        self.assertEqual(names, {"index_confluence_page", "search_confluence", "clear_index"})



if __name__ == "__main__":
    unittest.main()
