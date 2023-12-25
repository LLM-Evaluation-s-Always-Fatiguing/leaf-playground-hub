import json
from typing import Dict, Union, List

from leaf_playground.core.scene_definition.definitions.metric import _MetricData
from leaf_playground.core.workers.chart import Chart


class BarChart(Chart, chart_name="bar", supported_metric_names=["examinee.answer_question.accurate"]):

    def _generate(self, metrics: Dict[str, Union[_MetricData, List[_MetricData]]]):
        return json.loads("""{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {
    "values": [
      {"category":"A", "group": "x", "value":0.1},
      {"category":"A", "group": "y", "value":0.6},
      {"category":"A", "group": "z", "value":0.9},
      {"category":"B", "group": "x", "value":0.7},
      {"category":"B", "group": "y", "value":0.2},
      {"category":"B", "group": "z", "value":1.1},
      {"category":"C", "group": "x", "value":0.6},
      {"category":"C", "group": "y", "value":0.1},
      {"category":"C", "group": "z", "value":0.2}
    ]
  },
  "mark": "bar",
  "encoding": {
    "x": {"field": "category"},
    "y": {"field": "value", "type": "quantitative"},
    "xOffset": {"field": "group"},
    "color": {"field": "group"}
  }
}
""")


__all__ = ["BarChart"]
