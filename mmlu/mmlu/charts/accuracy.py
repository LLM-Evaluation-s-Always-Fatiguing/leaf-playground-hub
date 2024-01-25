from typing import List

import pandas as pd
from leaf_playground.chart_tools.simple_bar import SimpleBar
from leaf_playground.core.scene_definition import CombinedMetricsData, SceneConfig
from leaf_playground.core.workers import MetricEvaluatorConfig, Chart
from leaf_playground.data.log_body import LogBody


class AccuracyChart(Chart, chart_name="accuracy", supported_metric_names=["examinee.answer.accurate"]):

    def _generate(
            self,
            metrics: CombinedMetricsData,
            scene_config: SceneConfig,
            evaluator_configs: List[MetricEvaluatorConfig],
            logs: List[LogBody]
    ) -> dict:
        chart = SimpleBar(mode="percent")

        role_config = scene_config.roles_config.get_role_config('examinee')
        color_mapping = {
            agent_config.config_data.get('profile').get('id'): agent_config.config_data.get('chart_major_color') for
            agent_config in role_config.agents_config
        }

        data = self._transform_data(metrics['merged_metrics'], color_mapping)

        return chart.generate(data)

    @staticmethod
    def _transform_data(metrics, color_mapping: dict):
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

        flattened_results_pythonic = [
            {"agent": agent, "color": color_mapping.get(agent), "value": value}
            for agent, metrics in combined_results.items()
            for metric, value in metrics.items()
            if metric == 'accurate'
        ]

        return pd.DataFrame(flattened_results_pythonic)


__all__ = ["AccuracyChart"]
