from Benchmark.alns_benchmark import DUMMY_VEHICLE_NAME


def get_unassigned(self):
        """Get customers in dummy route (last route)."""
        if self.routes and self.vehicles[-1] == DUMMY_VEHICLE_NAME:
            return self.routes[-1]
        return []