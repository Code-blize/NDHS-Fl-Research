import numpy as np
import pytest
from flwr.common import ArrayRecord, Message, MetricRecord, RecordDict
from ndhs_flower.strategy import SixRegionFedAvg
from flwr.supercore.task_identity import TaskIdentity


def _reply(region, partition, n=10):
    TaskIdentity.task_id = 1
    TaskIdentity.run_id = 1
    TaskIdentity.node_id = 1

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord([
                np.array([float(region)], dtype=np.float32)
            ]),
            "metrics": MetricRecord({
                "num-examples": n,
                "region-code": region,
                "partition-id": partition,
                "unique-source-ids": n,
                "train-loss": 0.5,
            }),
        }),
        dst_node_id=1,
        message_type="train",
    )


def _strategy():
    return SixRegionFedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=6,
        min_available_nodes=6,
        weighted_by_key="num-examples",
    )


def test_strategy_contract_and_six_region_aggregation():
    strategy = _strategy()
    replies = [_reply(region, region - 1) for region in range(1, 7)]

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is not None
    assert metrics is not None
    assert strategy.round_audit[0]["regions"] == [1, 2, 3, 4, 5, 6]


def test_strategy_rejects_missing_region():
    strategy = _strategy()
    replies = [_reply(region, region - 1) for region in range(1, 6)]

    with pytest.raises(RuntimeError, match="exactly 6"):
        strategy.aggregate_train(1, replies)


def test_strategy_rejects_partition_region_mismatch():
    strategy = _strategy()
    replies = [_reply(region, region - 1) for region in range(1, 6)]
    replies.append(_reply(6, 4))

    with pytest.raises(RuntimeError, match="expected"):
        strategy.aggregate_train(1, replies)
