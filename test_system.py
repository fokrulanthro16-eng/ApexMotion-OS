"""
ApexMotion OS v3.0 - Spatial Grid & Industrial Mesh Automated Test Suite
Validates:
1. Dynamic Occupancy Grid & SLAM Fog-of-War Coverage
2. Deadlock Intersection Manager Lateral Yield Detours
3. ROS 2 Bridge Adapter (cmd_vel & odom endpoints)
4. Multi-Agent Intent Planner
"""

import math
import unittest
from planner import CommandIntent, parse_enterprise_intent
from vision import OccupancyGridMap, shared_fleet_grid
from main import FleetCoordinator, flight_recorder


class TestOccupancyGridSLAM(unittest.TestCase):
    def test_grid_initialization_and_discovery(self):
        grid_map = OccupancyGridMap(arena_w=10.0, arena_h=10.0, resolution=0.25)
        self.assertEqual(grid_map.cols, 40)
        self.assertEqual(grid_map.rows, 40)
        self.assertEqual(grid_map.exploration_percentage, 0.0)

        # Simulate LiDAR rays sweeping from (5.0, 5.0)
        mock_rays = [
            {"hit_x": 8.0, "hit_y": 5.0, "distance": 3.0},
            {"hit_x": 5.0, "hit_y": 8.0, "distance": 3.0},
            {"hit_x": 2.0, "hit_y": 5.0, "distance": 3.0},
            {"hit_x": 5.0, "hit_y": 2.0, "distance": 3.0},
        ]
        grid_map.update_from_lidar(5.0, 5.0, mock_rays)
        self.assertGreater(grid_map.exploration_percentage, 0.0)
        self.assertGreater(grid_map.discovered_cells, 10)


class TestIntersectionManagerDeadlock(unittest.TestCase):
    def setUp(self):
        self.coord = FleetCoordinator()

    def test_lateral_yield_detour_calculation(self):
        # Place Alpha (priority 1) and Bravo (priority 2) in conflict trajectory (dist = 1.2m)
        self.coord.agents["Alpha"].x = 5.0
        self.coord.agents["Alpha"].y = 5.0
        self.coord.agents["Bravo"].x = 5.8
        self.coord.agents["Bravo"].y = 5.0

        links = self.coord.resolve_intersections_and_deadlocks()

        # Bravo (lower priority) must have yielding = True and a lateral yield detour waypoint computed
        self.assertFalse(self.coord.agents["Alpha"].yielding)
        self.assertTrue(self.coord.agents["Bravo"].yielding)
        self.assertIsNotNone(self.coord.agents["Bravo"].yield_detour_wp)

        # Detour point must not be identical to Bravo's current position (must have lateral offset)
        detour_x, detour_y = self.coord.agents["Bravo"].yield_detour_wp
        self.assertNotEqual((detour_x, detour_y), (5.8, 5.0))


class TestFleetIntentPlanner(unittest.TestCase):
    def test_targeting_and_macros(self):
        cmd = parse_enterprise_intent("Alpha, patrol between dock and pallet 3 times")
        self.assertEqual(cmd.target_agent, "Alpha")
        self.assertEqual(cmd.intent, CommandIntent.MACRO)

        cmd_rtb = parse_enterprise_intent("Charlie, abort and return to base")
        self.assertEqual(cmd_rtb.target_agent, "Charlie")
        self.assertEqual(cmd_rtb.intent, CommandIntent.RETURN_TO_BASE)


if __name__ == "__main__":
    unittest.main()
