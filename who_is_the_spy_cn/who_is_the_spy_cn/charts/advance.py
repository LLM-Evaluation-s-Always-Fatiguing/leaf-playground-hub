from typing import List

import pandas as pd
from leaf_playground.chart_tools.grouped_bar import GroupedBar
from leaf_playground.core.scene_definition import CombinedMetricsData, SceneConfig
from leaf_playground.core.workers import MetricEvaluatorConfig, Chart
from leaf_playground.data.log_body import LogBody

ROLE_NAME = "player"
ACTION_NAMES = ["describe_key", "predict_role"]

SUPPORT_METRIC_NAMES = ["伪装能力", "推理能力"]
SUPPORT_METRIC_FULL_NAMES = ["player.describe_key.伪装能力", "player.predict_role.推理能力"]
METRIC_WEIGHT_MAPPING = {
    "推理能力": 0.6,
    "伪装能力": 0.4
}


class AdvanceChart(Chart, chart_name="advance", supported_metric_names=SUPPORT_METRIC_FULL_NAMES):

    def _generate(
        self,
        metrics: CombinedMetricsData,
        scene_config: SceneConfig,
        evaluator_configs: List[MetricEvaluatorConfig],
        logs: List[LogBody]
    ) -> dict:

        chart = GroupedBar(mode="value", max_value=5)

        data = self._transform_data(metrics['merged_metrics'])

        return chart.generate(data)

    @staticmethod
    def _transform_data(metrics):
        combined_results = {}
        for metric, entries in metrics.items():
            metric_key = metric.split('.')[-1]

            for entry in entries:
                agent = entry.target_agent
                value = entry.value

                if agent not in combined_results:
                    combined_results[agent] = {}

                if metric_key not in combined_results[agent]:
                    combined_results[agent][metric_key] = 0

                combined_results[agent][metric_key] += value

        for agent in combined_results.keys():
            weighted_sum = 0
            for metric, value in combined_results[agent].items():
                weighted_sum += value * METRIC_WEIGHT_MAPPING[metric]
            combined_results[agent]['※综合得分'] = weighted_sum

        flattened_results_pythonic = [
            {"agent": agent, "metric": metric, "value": value}
            for agent, metrics in combined_results.items()
            for metric, value in metrics.items()
        ]
        return pd.DataFrame(flattened_results_pythonic)


__all__ = ["AdvanceChart"]
