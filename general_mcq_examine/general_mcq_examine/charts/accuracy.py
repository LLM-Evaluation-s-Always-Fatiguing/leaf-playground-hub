import pandas as pd
from leaf_playground.chart_tools.grouped_normalized_bar import GroupedNormalizedBar
from leaf_playground.core.workers import CombinedMetricsData
from leaf_playground.core.workers.chart import Chart


class AccuracyChart(Chart, chart_name="bar", supported_metric_names=["examinee.answer_question.accurate"]):

    def _generate(self, metrics: CombinedMetricsData):
        chart = GroupedNormalizedBar(mode="percent")


        data = self._transform_data(metrics['metrics'])

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

        flattened_results_pythonic = [
            {"agent": agent, "metric": metric, "value": value}
            for agent, metrics in combined_results.items()
            for metric, value in metrics.items()
        ]
        return pd.DataFrame(flattened_results_pythonic)


__all__ = ["AccuracyChart"]
