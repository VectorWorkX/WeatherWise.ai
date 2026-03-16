"""Static baseline ECharts templates — fallback when Pinecone is unavailable.

One minimal but valid template per chart type. These are NOT the production
templates (those live in Pinecone's visual_templates namespace). These exist
only to ensure charts render even when Pinecone is down or returns no match.

The full indexing pipeline (scripts/build_chart_templates.py) will regenerate
this file with richer templates once run. Until then, these baselines work.
"""

BASELINE_TEMPLATES: dict[str, dict] = {
    "line": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "tooltip": {"trigger": "axis"},
            "legend": {"top": "bottom", "textStyle": {"color": "#ccc"}},
            "grid": {"left": "3%", "right": "4%", "bottom": "12%", "containLabel": True},
            "xAxis": {"type": "category", "data": [], "axisLabel": {"color": "#999"}},
            "yAxis": {"type": "value", "axisLabel": {"color": "#999", "formatter": "{value} {UNIT}"}},
            "series": [{"type": "line", "smooth": True, "data": [], "name": ""}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": "xAxis.data",
            "y_axis_name_path": "yAxis.name",
            "series_paths": ["series[0].data"],
            "legend_path": "legend.data",
            "category_path": None,
        },
    },
    "bar": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "tooltip": {"trigger": "axis"},
            "legend": {"top": "bottom", "textStyle": {"color": "#ccc"}},
            "grid": {"left": "3%", "right": "4%", "bottom": "12%", "containLabel": True},
            "xAxis": {"type": "category", "data": [], "axisLabel": {"color": "#999"}},
            "yAxis": {"type": "value", "axisLabel": {"color": "#999", "formatter": "{value} {UNIT}"}},
            "series": [{"type": "bar", "data": [], "name": "", "barWidth": "60%"}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": "xAxis.data",
            "y_axis_name_path": "yAxis.name",
            "series_paths": ["series[0].data"],
            "legend_path": "legend.data",
            "category_path": None,
        },
    },
    "pie": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"top": "bottom", "textStyle": {"color": "#ccc"}},
            "series": [{"type": "pie", "radius": "60%", "data": [],
                        "label": {"formatter": "{b}: {d}%"},
                        "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}}}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": None,
            "y_axis_name_path": None,
            "series_paths": ["series[0].data"],
            "legend_path": None,
            "category_path": None,
        },
    },
    "donut": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"top": "bottom", "textStyle": {"color": "#ccc"}},
            "series": [{"type": "pie", "radius": ["40%", "70%"], "data": [],
                        "itemStyle": {"borderRadius": 10, "borderColor": "#1a1a2e", "borderWidth": 2},
                        "label": {"show": False, "position": "center"},
                        "emphasis": {"label": {"show": True, "fontSize": 20, "fontWeight": "bold"}}}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": None,
            "y_axis_name_path": None,
            "series_paths": ["series[0].data"],
            "legend_path": None,
            "category_path": None,
        },
    },
    "radar": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "tooltip": {},
            "legend": {"top": "bottom", "textStyle": {"color": "#ccc"}, "data": []},
            "radar": {"indicator": []},
            "series": [{"type": "radar", "data": []}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": None,
            "y_axis_name_path": None,
            "series_paths": ["series[0].data"],
            "legend_path": "legend.data",
            "category_path": "radar.indicator",
        },
    },
    "gauge": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "series": [{"type": "gauge", "min": 0, "max": 100,
                        "progress": {"show": True, "width": 18},
                        "axisLine": {"lineStyle": {"width": 18}},
                        "axisTick": {"show": False},
                        "splitLine": {"length": 15, "lineStyle": {"width": 2, "color": "#999"}},
                        "axisLabel": {"distance": 25, "color": "#999"},
                        "detail": {"valueAnimation": True, "formatter": "{value} {UNIT}", "fontSize": 24},
                        "data": []}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": None,
            "y_axis_name_path": None,
            "series_paths": ["series[0].data"],
            "legend_path": None,
            "category_path": None,
        },
    },
    "scatter": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "tooltip": {"trigger": "item"},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {"type": "value", "axisLabel": {"color": "#999"}},
            "yAxis": {"type": "value", "axisLabel": {"color": "#999"}},
            "series": [{"type": "scatter", "data": [], "symbolSize": 10}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": None,
            "y_axis_name_path": "yAxis.name",
            "series_paths": ["series[0].data"],
            "legend_path": None,
            "category_path": None,
        },
    },
    "heatmap": {
        "echarts_template": {
            "title": {"text": "", "left": "center", "textStyle": {"color": "#ccc"}},
            "tooltip": {"position": "top"},
            "grid": {"left": "3%", "right": "8%", "bottom": "12%", "containLabel": True},
            "xAxis": {"type": "category", "data": [], "splitArea": {"show": True}},
            "yAxis": {"type": "category", "data": [], "splitArea": {"show": True}},
            "visualMap": {"min": 0, "max": 100, "calculable": True, "orient": "horizontal",
                          "left": "center", "bottom": "0%",
                          "inRange": {"color": ["#313695", "#4575b4", "#74add1", "#abd9e9",
                                                "#fee090", "#fdae61", "#f46d43", "#d73027"]}},
            "series": [{"type": "heatmap", "data": [], "label": {"show": True}}],
        },
        "data_mapping": {
            "title_path": "title.text",
            "x_axis_path": "xAxis.data",
            "y_axis_name_path": None,
            "series_paths": ["series[0].data"],
            "legend_path": None,
            "category_path": None,
        },
    },
}
