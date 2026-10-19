from __future__ import annotations

import unittest

from maya.framework.conversation.text import strip_citations_and_references


class SupervisorUtilsTests(unittest.TestCase):
    def test_preferences_is_not_mistaken_for_references_heading(self) -> None:
        text = "The choice may depend on individual preferences and physical ability."

        self.assertEqual(strip_citations_and_references(text), text)

    def test_references_in_normal_prose_is_preserved(self) -> None:
        text = "The article references several earlier studies."

        self.assertEqual(strip_citations_and_references(text), text)

    def test_actual_references_section_is_removed(self) -> None:
        text = "Answer sentence [1].\n\nReferences:\n[1] Example paper."

        self.assertEqual(strip_citations_and_references(text), "Answer sentence .")

    def test_disclaimer_after_references_is_retained(self) -> None:
        text = (
            "Answer sentence.\n\n# References\n[1] Example paper.\n\n"
            "Safety Disclaimer\nSeek urgent medical help when appropriate."
        )

        self.assertEqual(
            strip_citations_and_references(text),
            "Answer sentence.\n\nSafety Disclaimer\nSeek urgent medical help when appropriate.",
        )


if __name__ == "__main__":
    unittest.main()
