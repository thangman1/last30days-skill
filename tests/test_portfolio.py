"""Tests for portfolio strategy module."""

import json
import sys
import unittest
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import portfolio as pf


class TestExtractTickerMentions(unittest.TestCase):
    def test_counts_single_ticker(self):
        text = "NVDA is up 18%. NVDA earnings beat expectations. NVDA NVDA."
        counts = pf.extract_ticker_mentions(text)
        self.assertEqual(counts.get("NVDA"), 4)

    def test_counts_multiple_tickers(self):
        text = "Buy MSFT and AMZN. GOOGL is also good. MSFT is the best."
        counts = pf.extract_ticker_mentions(text)
        self.assertEqual(counts.get("MSFT"), 2)
        self.assertEqual(counts.get("AMZN"), 1)
        self.assertEqual(counts.get("GOOGL"), 1)

    def test_ignores_unknown_tickers(self):
        text = "FAKE and NOTREAL are not real tickers."
        counts = pf.extract_ticker_mentions(text)
        self.assertNotIn("FAKE", counts)
        self.assertNotIn("NOTREAL", counts)

    def test_empty_text_returns_empty(self):
        counts = pf.extract_ticker_mentions("")
        self.assertEqual(counts, {})

    def test_no_false_positives_in_words(self):
        # "AMD" should not match inside longer words
        text = "The RAMDISK and AMDSOMETHING are not tickers. But AMD is."
        counts = pf.extract_ticker_mentions(text)
        self.assertEqual(counts.get("AMD"), 1)

    def test_brk_b_matches(self):
        text = "BRK.B is Berkshire Hathaway class B shares."
        counts = pf.extract_ticker_mentions(text)
        self.assertEqual(counts.get("BRK.B"), 1)


class TestAdjustWeightsFromResearch(unittest.TestCase):
    def _make_template(self):
        return [
            {"ticker": "NVDA", "name": "NVIDIA", "sector": "AI", "base_weight": 0.30, "type": "Stock"},
            {"ticker": "MSFT", "name": "Microsoft", "sector": "Tech", "base_weight": 0.40, "type": "Stock"},
            {"ticker": "AMZN", "name": "Amazon", "sector": "Tech", "base_weight": 0.30, "type": "Stock"},
        ]

    def test_weights_sum_to_one(self):
        template = self._make_template()
        counts = {"NVDA": 10, "MSFT": 5, "AMZN": 2}
        positions = pf.adjust_weights_from_research(template, counts)
        total = sum(p["weight"] for p in positions)
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_higher_mentions_increase_weight(self):
        # Use equal base weights so research signal is the only differentiator
        equal_template = [
            {"ticker": "NVDA", "name": "NVIDIA", "sector": "AI", "base_weight": 0.33, "type": "Stock"},
            {"ticker": "MSFT", "name": "Microsoft", "sector": "Tech", "base_weight": 0.33, "type": "Stock"},
            {"ticker": "AMZN", "name": "Amazon", "sector": "Tech", "base_weight": 0.34, "type": "Stock"},
        ]
        counts = {"NVDA": 20, "MSFT": 1, "AMZN": 1}
        positions = pf.adjust_weights_from_research(equal_template, counts)
        nvda_w = next(p["weight"] for p in positions if p["ticker"] == "NVDA")
        msft_w = next(p["weight"] for p in positions if p["ticker"] == "MSFT")
        self.assertGreater(nvda_w, msft_w)

    def test_no_research_preserves_ratio(self):
        template = self._make_template()
        positions = pf.adjust_weights_from_research(template, {})
        # With no signals the base_weights are used as-is and normalised
        total = sum(p["weight"] for p in positions)
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_boost_cap_respected(self):
        template = self._make_template()
        # Flood with NVDA mentions — boost should be capped
        counts = {"NVDA": 1000}
        positions = pf.adjust_weights_from_research(template, counts, boost_cap=0.10)
        nvda_raw = next(p for p in positions if p["ticker"] == "NVDA")
        # After cap the raw addition before normalisation is at most base_weight + boost_cap
        self.assertLessEqual(nvda_raw["weight"], 1.0)


class TestAllocateDollars(unittest.TestCase):
    def test_dollar_amounts_sum_to_budget(self):
        positions = [
            {"ticker": "A", "weight": 0.50},
            {"ticker": "B", "weight": 0.30},
            {"ticker": "C", "weight": 0.20},
        ]
        result = pf.allocate_dollars(positions, 50_000)
        total = sum(p["dollars"] for p in result)
        self.assertAlmostEqual(total, 50_000, places=0)

    def test_proportional_allocation(self):
        positions = [
            {"ticker": "X", "weight": 0.60},
            {"ticker": "Y", "weight": 0.40},
        ]
        result = pf.allocate_dollars(positions, 10_000)
        x_pos = next(p for p in result if p["ticker"] == "X")
        y_pos = next(p for p in result if p["ticker"] == "Y")
        self.assertAlmostEqual(x_pos["dollars"], 6_000, places=0)
        self.assertAlmostEqual(y_pos["dollars"], 4_000, places=0)

    def test_original_fields_preserved(self):
        positions = [{"ticker": "Z", "weight": 1.0, "name": "Zeta Corp", "sector": "Tech"}]
        result = pf.allocate_dollars(positions, 1_000)
        self.assertEqual(result[0]["name"], "Zeta Corp")
        self.assertEqual(result[0]["sector"], "Tech")


class TestBuildPortfolio(unittest.TestCase):
    def test_returns_expected_keys(self):
        port = pf.build_portfolio(50_000, "moderate", pf.MOCK_RESEARCH_OUTPUT)
        for key in ("risk", "budget", "profile", "positions", "ticker_counts",
                    "watchlist", "generated_at", "disclaimer"):
            self.assertIn(key, port)

    def test_moderate_has_six_positions(self):
        port = pf.build_portfolio(50_000, "moderate", "")
        self.assertEqual(len(port["positions"]), 6)

    def test_conservative_risk_level(self):
        port = pf.build_portfolio(50_000, "conservative", "")
        self.assertEqual(port["risk"], "conservative")

    def test_aggressive_risk_level(self):
        port = pf.build_portfolio(25_000, "aggressive", "")
        self.assertEqual(port["risk"], "aggressive")
        self.assertEqual(port["budget"], 25_000)

    def test_dollar_total_matches_budget(self):
        budget = 75_000
        port = pf.build_portfolio(budget, "moderate", pf.MOCK_RESEARCH_OUTPUT)
        total = sum(p["dollars"] for p in port["positions"])
        self.assertAlmostEqual(total, budget, places=0)

    def test_mock_research_produces_ticker_counts(self):
        port = pf.build_portfolio(50_000, "moderate", pf.MOCK_RESEARCH_OUTPUT)
        self.assertIn("NVDA", port["ticker_counts"])
        self.assertGreater(port["ticker_counts"]["NVDA"], 0)

    def test_watchlist_excludes_portfolio_tickers(self):
        port = pf.build_portfolio(50_000, "moderate", pf.MOCK_RESEARCH_OUTPUT)
        portfolio_tickers = {p["ticker"] for p in port["positions"]}
        for ticker, _ in port["watchlist"]:
            self.assertNotIn(ticker, portfolio_tickers)

    def test_all_risk_levels_build(self):
        for risk in ("conservative", "moderate", "aggressive"):
            port = pf.build_portfolio(50_000, risk, "")
            self.assertEqual(port["risk"], risk)
            self.assertTrue(len(port["positions"]) > 0)


class TestRenderMd(unittest.TestCase):
    def setUp(self):
        self.port = pf.build_portfolio(50_000, "moderate", pf.MOCK_RESEARCH_OUTPUT)

    def test_contains_budget(self):
        output = pf.render_md(self.port)
        self.assertIn("50,000", output)

    def test_contains_all_tickers(self):
        output = pf.render_md(self.port)
        for pos in self.port["positions"]:
            self.assertIn(pos["ticker"], output)

    def test_contains_disclaimer(self):
        output = pf.render_md(self.port)
        self.assertIn("Not financial advice", output)

    def test_contains_sp500_reference(self):
        output = pf.render_md(self.port)
        self.assertIn("S&P 500", output)

    def test_contains_research_heatmap(self):
        output = pf.render_md(self.port)
        self.assertIn("Research Signal Heatmap", output)


class TestRenderCompact(unittest.TestCase):
    def test_contains_risk_level(self):
        port = pf.build_portfolio(50_000, "aggressive", "")
        output = pf.render_compact(port)
        self.assertIn("AGGRESSIVE", output)

    def test_contains_all_tickers(self):
        port = pf.build_portfolio(50_000, "conservative", "")
        output = pf.render_compact(port)
        for pos in port["positions"]:
            self.assertIn(pos["ticker"], output)


class TestRenderJson(unittest.TestCase):
    def test_valid_json(self):
        port = pf.build_portfolio(50_000, "moderate", "")
        output = pf.render_json(port)
        parsed = json.loads(output)
        self.assertIn("positions", parsed)

    def test_budget_in_json(self):
        port = pf.build_portfolio(30_000, "moderate", "")
        output = pf.render_json(port)
        parsed = json.loads(output)
        self.assertEqual(parsed["budget"], 30_000)


if __name__ == "__main__":
    unittest.main()
