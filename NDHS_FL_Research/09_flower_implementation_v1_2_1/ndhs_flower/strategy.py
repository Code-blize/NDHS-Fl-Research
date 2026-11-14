from flwr.serverapp.strategy import FedAvg

class SixRegionFedAvg(FedAvg):
    """FedAvg with public-message-contract checks for exactly six regional replies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_audit = []

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        if len(replies) != 6:
            raise RuntimeError(
                f"Round {server_round}: expected exactly 6 replies, received {len(replies)}"
            )

        valid = []
        errors = []
        for message in replies:
            if message.has_error():
                reason = message.error.reason if message.error is not None else "unspecified client error"
                errors.append(reason)
            else:
                valid.append(message)

        if errors:
            raise RuntimeError(f"Round {server_round}: client errors: {errors}")
        if len(valid) != 6:
            raise RuntimeError(f"Round {server_round}: expected 6 valid replies, got {len(valid)}")

        observed = []
        for message in valid:
            content = message.content
            if "arrays" not in content or "metrics" not in content:
                raise RuntimeError(f"Round {server_round}: reply missing arrays or metrics")
            metrics = content["metrics"]
            required = {"num-examples", "region-code", "partition-id", "unique-source-ids", "train-loss"}
            missing = required.difference(metrics.keys())
            if missing:
                raise RuntimeError(f"Round {server_round}: reply missing metrics {sorted(missing)}")
            region = int(metrics["region-code"])
            partition = int(metrics["partition-id"])
            n_examples = int(metrics["num-examples"])
            n_unique = int(metrics["unique-source-ids"])
            if region != partition + 1:
                raise RuntimeError(
                    f"Round {server_round}: partition {partition} reported region {region}; expected {partition + 1}"
                )
            if n_examples <= 0 or n_unique != n_examples:
                raise RuntimeError(
                    f"Round {server_round}: invalid example/unique-ID counts for region {region}: "
                    f"{n_examples}/{n_unique}"
                )
            observed.append((region, partition, n_examples, n_unique))

        regions = [item[0] for item in observed]
        if sorted(regions) != [1, 2, 3, 4, 5, 6]:
            raise RuntimeError(f"Round {server_round}: expected regions 1..6 exactly once, got {regions}")

        aggregated_arrays, aggregated_metrics = super().aggregate_train(server_round, valid)
        if aggregated_arrays is None:
            raise RuntimeError(f"Round {server_round}: FedAvg returned no aggregated arrays")
        self.round_audit.append({
            "server_round": int(server_round),
            "clients_expected": 6,
            "clients_replied": len(valid),
            "regions": sorted(regions),
            "client_participation": [
                {"region_code": r, "partition_id": p, "num_examples": n, "unique_source_ids": u}
                for r, p, n, u in sorted(observed)
            ],
        })
        return aggregated_arrays, aggregated_metrics
