import unittest
from types import SimpleNamespace

from v2.execution import OrderPlan, build_order_plans, place_orders_with_limits


class _TimeoutClient:
    def __init__(self):
        self.orders = {}
        self.next_id = 1

    def submit_order(self, request):
        oid = str(self.next_id)
        self.next_id += 1
        is_limit = getattr(request, "limit_price", None) is not None
        qty = float(getattr(request, "qty", 0.0))
        if is_limit:
            self.orders[oid] = {"status": "new", "filled_qty": 0.0, "qty": qty, "type": "limit"}
        else:
            self.orders[oid] = {"status": "filled", "filled_qty": qty, "qty": qty, "type": "market"}
        return SimpleNamespace(id=oid)

    def get_order_by_id(self, order_id: str):
        record = self.orders.get(order_id, {"status": "canceled", "filled_qty": 0.0, "qty": 0.0})
        return SimpleNamespace(id=order_id, status=record["status"], filled_qty=record.get("filled_qty", 0.0), qty=record.get("qty", 0.0))

    def cancel_order_by_id(self, order_id: str):
        if order_id in self.orders:
            self.orders[order_id]["status"] = "canceled"


class ExecutionTests(unittest.TestCase):
    def test_build_order_plans_respects_cost_inputs(self):
        plans = build_order_plans(
            targets={"SPY": 100000.0},
            current_mv={"SPY": 0.0},
            prices={"SPY": 100.0},
            adv={"SPY": 200000.0},
            min_notional=1000.0,
            max_slices=10,
            spread_bps=5.0,
            kappa=0.5,
            psi=0.5,
        )
        self.assertTrue(plans, "Expected at least one order plan")
        self.assertGreaterEqual(plans[0].slices, 2, "Cost model should increase slices beyond minimum")
        self.assertIsNotNone(plans[0].limit_price)

    def test_place_orders_timeout_falls_back_to_market(self):
        client = _TimeoutClient()
        plan = OrderPlan(symbol="SPY", side="buy", qty=10.0, limit_price=101.0, slices=1)
        last_prices = {"SPY": 100.0}
        ids = place_orders_with_limits(client, [plan], last_prices, peg_bps=5, fill_timeout_sec=0)
        # Expect the original limit id and the fallback market id
        self.assertEqual(len(ids), 2)
        self.assertIn("1", ids)
        self.assertIn("2", ids)
        self.assertEqual(client.orders["1"]["status"], "canceled")
        self.assertEqual(client.orders["2"]["status"], "filled")


if __name__ == "__main__":
    unittest.main()
